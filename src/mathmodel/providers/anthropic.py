"""MathModel AI — Anthropic (Claude) provider implementation."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional, Type

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from mathmodel.providers.base import (
    BaseModelProvider,
    GenerationRequest,
    GenerationResponse,
    StructuredGenerationRequest,
    UsageInfo,
)


class AnthropicProvider(BaseModelProvider):
    """Anthropic Claude model provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-sonnet-4-20250514",
    ):
        super().__init__(api_key=api_key, default_model=default_model)
        self._client: Optional[AsyncAnthropic] = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            if not self.api_key:
                raise ValueError("Anthropic API key is required")
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        client = self._get_client()
        model = request.model or self.default_model

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        if request.temperature > 0:
            kwargs["temperature"] = request.temperature
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences

        response = await client.messages.create(**kwargs)

        # Extract text content
        content_blocks = response.content
        text = ""
        for block in content_blocks:
            if hasattr(block, "text"):
                text += block.text

        usage = UsageInfo(
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
            total_tokens=(
                (response.usage.input_tokens + response.usage.output_tokens)
                if response.usage
                else 0
            ),
        )
        self._record_usage(usage)

        return GenerationResponse(
            content=text,
            model=response.model,
            usage=usage,
            finish_reason=response.stop_reason or "stop",
            is_mock=False,
        )

    async def structured_generate(
        self, request: StructuredGenerationRequest
    ) -> BaseModel:
        # Claude doesn't have native structured output like OpenAI
        # Fall back to generate + parse with JSON instructions
        schema_json = request.output_schema.model_json_schema()
        enhanced_prompt = (
            f"{request.prompt}\n\n"
            f"You MUST respond with ONLY a valid JSON object. "
            f"The JSON must conform to this schema:\n"
            f"{json.dumps(schema_json, indent=2)}\n\n"
            f"Do not include any explanation, markdown formatting, or code fences. "
            f"Output ONLY the JSON object."
        )

        gen_request = GenerationRequest(
            prompt=enhanced_prompt,
            system_prompt=request.system_prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=0.0,  # Lower temperature for structured output
        )

        response = await self.generate(gen_request)
        content = response.content.strip()

        # Remove markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:]  # Remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove closing fence
            content = "\n".join(lines)

        parsed = json.loads(content)
        return request.output_schema.model_validate(parsed)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        client = self._get_client()
        model = request.model or self.default_model

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        if request.temperature > 0:
            kwargs["temperature"] = request.temperature

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text