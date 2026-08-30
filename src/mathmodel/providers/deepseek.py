"""MathModel AI — DeepSeek provider via OpenAI-compatible API.

Many DeepSeek deployments expose an OpenAI-compatible endpoint.
This provider reuses the OpenAI client with DeepSeek defaults.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from mathmodel.providers.base import (
    BaseModelProvider,
    GenerationRequest,
    GenerationResponse,
    StructuredGenerationRequest,
    UsageInfo,
)


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
            )
        return self._client

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        client = self._get_client()
        messages = self._build_messages(request.prompt, request.system_prompt)
        model = request.model or self.default_model

        response = await client.chat.completions.create(
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
        model = request.model or self.default_model

        # DeepSeek (OpenAI-compatible) requires the word "json" in the
        # prompt for response_format=json_object. Append the schema to
        # the prompt so the model knows the expected JSON shape. Keep the
        # FULL schema (required fields included) — the compact form caused
        # the model to omit required nested fields.
        schema_json = json.dumps(request.output_schema.model_json_schema())
        enhanced_prompt = (
            f"{request.prompt}\n\n"
            f"You must respond with ONLY a valid JSON object conforming "
            f"to this JSON schema:\n{schema_json}\n"
            f"Do not include any explanation or markdown fences. "
            f"All fields marked 'required' MUST be present."
        )

        messages = self._build_messages(enhanced_prompt, request.system_prompt)

        # Large structured schemas need headroom: avoid finish_reason=length
        max_tokens = max(request.max_tokens, 16384)

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=request.temperature,
            response_format={"type": "json_object"},
        )

        choice = response.choices[0]
        content = choice.message.content or "{}"
        usage = UsageInfo(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )
        self._record_usage(usage)

        parsed = json.loads(content, strict=False)
        return request.output_schema.model_validate(parsed)

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
        """Reduce a JSON schema to field names + types to keep prompts small.

        Large nested schemas (e.g. ProblemAnalysis) balloon the prompt and
        cause the model to hit max_tokens mid-JSON (finish_reason=length).
        This keeps structure but drops descriptions/defaults/examples.
        """
        def strip(node):
            if isinstance(node, dict):
                out = {}
                for key in ("type", "properties", "items", "required"):
                    if key in node:
                        out[key] = strip(node[key])
                if "anyOf" in node:
                    out["anyOf"] = [strip(o) for o in node["anyOf"]]
                if "enum" in node:
                    out["enum"] = node["enum"]
                return out
            if isinstance(node, list):
                return [strip(x) for x in node]
            return node

        return strip(schema)