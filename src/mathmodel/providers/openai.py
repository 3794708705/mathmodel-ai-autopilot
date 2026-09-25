"""MathModel AI — OpenAI provider implementation.

Also serves any OpenAI-compatible gateway: set OPENAI_BASE_URL (and
OPENAI_API_KEY) to point at a local or self-hosted endpoint.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from mathmodel.config import get_settings
from mathmodel.providers.base import (
    BaseModelProvider,
    GenerationRequest,
    GenerationResponse,
    StructuredGenerationRequest,
    UsageInfo,
)
from mathmodel.providers.json_repair import normalise_for_schema, parse_json_object

logger = logging.getLogger(__name__)

# Large domain schemas (MathematicalModel, PaperOutline) need far more room
# than the provider default, or the JSON body is cut off mid-object.
STRUCTURED_MAX_TOKENS = 32768


class OpenAIProvider(BaseModelProvider):
    """OpenAI model provider (GPT-4o, GPT-4.1, etc.), or any compatible gateway."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o",
        base_url: Optional[str] = None,
    ):
        super().__init__(api_key=api_key, default_model=default_model)
        self._client: Optional[AsyncOpenAI] = None
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("OpenAI API key is required")
            kwargs: dict[str, Any] = {"api_key": self.api_key, "max_retries": 5}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        client = self._get_client()
        messages = self._build_messages(request.prompt, request.system_prompt)
        model = request.model or self.default_model

        response = await self._chat_completion(
            client,
            reasoning_effort=request.reasoning_effort,
            extra_body=request.extra_body,
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

    async def structured_generate(
        self, request: StructuredGenerationRequest
    ) -> BaseModel:
        client = self._get_client()
        messages = self._build_messages(request.prompt, request.system_prompt)
        model = request.model or self.default_model

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
        max_tokens = max(request.max_tokens, STRUCTURED_MAX_TOKENS)

        async def _call(**overrides: Any):
            kwargs: dict[str, Any] = {
                "reasoning_effort": request.reasoning_effort,
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": request.temperature,
                "response_format": {"type": "json_object"},
            }
            kwargs.update(overrides)
            return await self._chat_completion(client, **kwargs)

        try:
            response = await _call()
        except Exception as exc:
            if "response_format" not in str(exc).lower() and "json_object" not in str(exc).lower():
                raise
            logger.warning("response_format=json_object rejected; retrying without it")
            response = await _call(response_format=None)

        # A truncated response is a wasted call: the JSON body is cut off
        # mid-object and can never be parsed. Gateways that bill reasoning
        # against the completion budget hit this often, so retry once with
        # reasoning switched off rather than letting the stage fail on a
        # response that was never going to be valid.
        if response.choices[0].finish_reason == "length":
            logger.warning(
                "structured response truncated at %s tokens; "
                "retrying with thinking disabled",
                max_tokens,
            )
            try:
                response = await _call(
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except Exception as exc:  # gateway may not support `thinking`
                logger.warning("thinking-disabled retry failed (%s); keeping first response", exc)

        choice = response.choices[0]
        content = choice.message.content or "{}"
        usage = UsageInfo(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )
        self._record_usage(usage)

        parsed = parse_json_object(content)
        if not parsed:
            # Truncated or empty bodies are usually transient; retry once with a
            # larger budget before failing the whole pipeline stage.
            logger.warning(
                "Structured response was not usable (%d chars, finish_reason=%s); retrying",
                len(content), choice.finish_reason,
            )
            repair = await self._chat_completion(
                client,
                reasoning_effort=request.reasoning_effort,
                model=model,
                messages=self._build_messages(
                    f"{enhanced_prompt}\n\nYour previous response was not a usable "
                    f"JSON object. Respond again with a SINGLE compact JSON object "
                    f"matching the schema. Keep every string short, never repeat the "
                    f"schema, and always emit the closing brace.",
                    request.system_prompt,
                ),
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
            parsed = parse_json_object(content)
            if not parsed:
                raise ValueError(
                    f"Provider returned no usable JSON after retry "
                    f"({len(content)} chars): {content[:400]}"
                )

        return request.output_schema.model_validate(
            normalise_for_schema(parsed, request.output_schema)
        )

    async def _chat_completion(self, client, reasoning_effort=None, extra_body=None, **kwargs):
        """Call the chat API with a bounded reasoning budget when supported."""
        if extra_body:
            kwargs["extra_body"] = extra_body
        effort = reasoning_effort or get_settings().deepseek_reasoning_effort
        if not effort:
            return await self._call(client, kwargs)
        try:
            return await self._call(client, kwargs, reasoning_effort=effort)
        except Exception as exc:
            if "reasoning_effort" not in str(exc).lower():
                raise
            logger.warning(
                "Provider rejected reasoning_effort=%s; retrying without it", effort
            )
            return await self._call(client, kwargs)

    @staticmethod
    async def _call(client, kwargs: dict, **extra):
        try:
            return await client.chat.completions.create(**kwargs, **extra)
        except Exception as exc:
            # An unsupported extra_body key must not fail the whole request.
            if "extra_body" not in kwargs or "extra_body" in str(exc).lower():
                raise
            retry = {k: v for k, v in kwargs.items() if k != "extra_body"}
            logger.warning("Provider rejected extra_body; retrying without it")
            return await client.chat.completions.create(**retry, **extra)

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