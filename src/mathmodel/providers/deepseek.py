"""MathModel AI — DeepSeek provider via OpenAI-compatible API.

Many DeepSeek deployments expose an OpenAI-compatible endpoint.
This provider reuses the OpenAI client with DeepSeek defaults.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from mathmodel.config import get_settings
from mathmodel.providers.base import (
    BaseModelProvider,
    GenerationRequest,
    GenerationResponse,
    StructuredGenerationRequest,
    UsageInfo,
)


from mathmodel.providers.json_repair import (
    coerce_enum_casing as _coerce_enum_casing,
    normalise_for_schema as _normalise_for_schema,
    outermost_object as _outermost_object,
    parse_json_object,
    resolve_schema as _resolve_schema,
)


# Structured outputs for large domain schemas (MathematicalModel, PaperOutline,
# ProblemAnalysis) routinely exceed a few thousand tokens; the API accepts far
# more, so give them room instead of silently truncating the JSON body.
STRUCTURED_MAX_TOKENS = 32768


class DeepSeekProvider(BaseModelProvider):
    """DeepSeek model provider (OpenAI-compatible endpoint).

    Defaults to DEEPSEEK_API_KEY env var.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "deepseek-chat",
        base_url: Optional[str] = None,
    ):
        super().__init__(api_key=api_key, default_model=default_model)
        self._client: Optional[AsyncOpenAI] = None
        self._base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("DEEPSEEK_API_KEY is required")
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self._base_url,
                max_retries=5,
            )
        return self._client

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        client = self._get_client()
        messages = self._build_messages(request.prompt, request.system_prompt)
        model = request.model or self.default_model

        response = await self._chat_completion(
            client,
            reasoning_effort=request.reasoning_effort,
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop_sequences,
        )

        choice = response.choices[0]
        usage = UsageInfo(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )
        self._record_usage(usage)

        return GenerationResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
            is_mock=False,
        )

    async def _chat_completion(self, client, reasoning_effort: Optional[str] = None, **kwargs):
        """Call the chat API with a bounded reasoning budget.

        These models otherwise spend the entire output budget on reasoning and
        return an empty or truncated answer (measured: 32039 of 32768
        completion tokens counted as reasoning, finish_reason=length, 2296
        characters of text). Bounding the budget leaves room for the actual
        answer. Deployments that reject the parameter fall back cleanly.
        """
        effort = reasoning_effort or get_settings().deepseek_reasoning_effort
        if not effort:
            return await client.chat.completions.create(**kwargs)
        try:
            return await client.chat.completions.create(
                reasoning_effort=effort, **kwargs
            )
        except Exception as exc:
            if "reasoning_effort" not in str(exc).lower():
                raise
            logger.warning(
                "Provider rejected reasoning_effort=%s; retrying without it", effort
            )
            return await client.chat.completions.create(**kwargs)

    async def structured_generate(
        self, request: StructuredGenerationRequest
    ) -> BaseModel:
        client = self._get_client()
        model = request.model or self.default_model

        # The reasoning model spends its whole output budget on reasoning and
        # returns no JSON body at all for large schemas (measured: 16384/32768
        # completion tokens all counted as reasoning, finish_reason=length,
        # empty content). Structured extraction therefore uses the standard
        # model; the reasoning model stays available for free-text generation.
        if model == get_settings().deepseek_reasoning_model:
            logger.warning(
                "Structured generation cannot use the reasoning model %s "
                "(it consumes the full output budget on reasoning and returns "
                "no JSON); falling back to %s",
                model, self.default_model,
            )
            model = self.default_model

        # Use full schema. Native DeepSeek handles large prompts well.
        schema_json = json.dumps(request.output_schema.model_json_schema())
        enhanced_prompt = (
            f"{request.prompt}\n\n"
            f"You must respond with ONLY a valid JSON object conforming "
            f"to this JSON schema:\n{schema_json}\n"
            f"Do not include any explanation or markdown fences. "
            f"All fields marked 'required' MUST be present. "
            f"Produce an INSTANCE that satisfies the schema; never repeat the "
            f"schema itself."
        )

        messages = self._build_messages(enhanced_prompt, request.system_prompt)

        # Large structured schemas (a full MathematicalModel can easily exceed
        # 5k characters) must not be cut off by the output cap: a truncated
        # body is unparseable JSON and would fail a whole pipeline stage.
        max_tokens = max(request.max_tokens, STRUCTURED_MAX_TOKENS)

        # Try with response_format first; fall back to prompt-only JSON
        # if the provider rejects json_object mode (some API gateways).
        try:
            response = await self._chat_completion(
                client,
                reasoning_effort=request.reasoning_effort,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            msg = str(e)
            if "403" in msg or "denied" in msg.lower() or "json_object" in msg.lower() or "response_format" in msg.lower():
                logger.warning(
                    "response_format=json_object rejected (403/denied), falling back to prompt-only JSON"
                )
                response = await self._chat_completion(
                    client,
                    reasoning_effort=request.reasoning_effort,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=request.temperature,
                )
            else:
                raise

        choice = response.choices[0]
        content = choice.message.content or "{}"
        usage = UsageInfo(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )
        self._record_usage(usage)

        parsed = self._parse_json(content)
        if not parsed:
            # A truncated, malformed or empty JSON body is a real failure, but
            # it is usually transient. Retry once with a bigger budget and an
            # explicit instruction before giving up, so one bad response cannot
            # fail a whole pipeline stage.
            truncated = choice.finish_reason == "length"
            logger.warning(
                "Structured response was not usable (%d chars, finish_reason=%s, "
                "reasoning_tokens=%s); retrying",
                len(content), choice.finish_reason,
                getattr(getattr(response.usage, "completion_tokens_details", None),
                        "reasoning_tokens", None),
            )
            repair_prompt = (
                f"{enhanced_prompt}\n\n"
                f"Your previous response was not a usable JSON object"
                + (" because it was cut off before the closing brace." if truncated else ".")
                + " Respond again with a SINGLE compact JSON object matching the schema."
                " Keep every string short (prefer under 120 characters), include only"
                " the most important items, and never repeat the schema or the prompt."
                " Do not use markdown fences, comments, or trailing commas."
                " Always emit the closing brace."
            )
            repair = await self._chat_completion(
                client,
                reasoning_effort=request.reasoning_effort,
                model=model,
                messages=self._build_messages(repair_prompt, request.system_prompt),
                max_tokens=STRUCTURED_MAX_TOKENS * 2,
                temperature=request.temperature,
                response_format={"type": "json_object"},
            )
            content = repair.choices[0].message.content or "{}"
            if repair.usage:
                self._record_usage(UsageInfo(
                    prompt_tokens=repair.usage.prompt_tokens,
                    completion_tokens=repair.usage.completion_tokens,
                    total_tokens=repair.usage.total_tokens,
                ))
            parsed = self._parse_json(content)
            if not parsed:
                raise ValueError(
                    f"Provider returned no usable JSON after retry "
                    f"({len(content)} chars): {content[:400]}"
                )

        return request.output_schema.model_validate(
            _normalise_for_schema(parsed, request.output_schema)
        )

    @staticmethod
    def _parse_json(content: str) -> Optional[dict]:
        """Parse a JSON object from a model response, tolerating fences."""
        return parse_json_object(content)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        client = self._get_client()
        messages = self._build_messages(request.prompt, request.system_prompt)
        model = request.model or self.default_model

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _compact_schema(schema: dict) -> dict:
        """Reduce a JSON schema to keep prompt size manageable.

        Keeps field names, types, required markers, and enum values.
        Drops descriptions, defaults, examples, and deep nesting.
        Target: ~500-2000 chars for typical schemas.
        """
        def strip(node, depth=0):
            if isinstance(node, dict):
                if depth > 3:
                    return {"type": node.get("type", "string")}
                out = {}
                for key in ("type", "properties", "items", "required"):
                    if key in node:
                        if key == "properties" and depth >= 1:
                            props = {}
                            for pname, pval in node[key].items():
                                if isinstance(pval, dict):
                                    prop_type = pval.get("type", "string")
                                    entry = {"type": prop_type}
                                    if "enum" in pval:
                                        entry["enum"] = pval["enum"]
                                    if "items" in pval:
                                        entry["items"] = strip(pval["items"], depth + 2)
                                    props[pname] = entry
                                else:
                                    props[pname] = pval
                            out[key] = props
                        else:
                            out[key] = strip(node[key], depth + 1)
                if "anyOf" in node:
                    out["anyOf"] = [strip(o, depth + 1) for o in node["anyOf"]]
                if "enum" in node:
                    out["enum"] = node["enum"]
                if "additionalProperties" in node:
                    out["additionalProperties"] = node["additionalProperties"]
                return out
            if isinstance(node, list):
                return [strip(x, depth) for x in node]
            return node

        return strip(schema)