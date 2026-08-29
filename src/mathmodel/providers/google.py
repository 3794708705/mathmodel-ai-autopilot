"""MathModel AI — Google (Gemini) provider implementation."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional, Type

from pydantic import BaseModel

from mathmodel.providers.base import (
    BaseModelProvider,
    GenerationRequest,
    GenerationResponse,
    StructuredGenerationRequest,
    UsageInfo,
)


class GoogleProvider(BaseModelProvider):
    """Google Gemini model provider.

    Requires google-generativeai package. Falls back to MockProvider
    behavior if the package is not installed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.0-flash",
    ):
        super().__init__(api_key=api_key, default_model=default_model)
        self._client = None
        self._available = self._check_availability()

    @staticmethod
    def _check_availability() -> bool:
        try:
            import google.generativeai  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def provider_name(self) -> str:
        return "google"

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("Google API key is required")
            if not self._available:
                raise ImportError(
                    "google-generativeai package is not installed. "
                    "Install with: pip install google-generativeai"
                )
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai
        return self._client

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._available:
            return GenerationResponse(
                content="[UNAVAILABLE] google-generativeai package not installed.",
                model=request.model or self.default_model,
                is_mock=True,
            )
        client = self._get_client()
        model_name = request.model or self.default_model
        model = client.GenerativeModel(model_name)

        contents = []
        if request.system_prompt:
            contents.append({"role": "user", "parts": [request.system_prompt]})
            contents.append({"role": "model", "parts": ["Understood."]})
        contents.append({"role": "user", "parts": [request.prompt]})

        response = await model.generate_content_async(
            contents,
            generation_config={
                "max_output_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
        )

        text = response.text or ""
        usage = UsageInfo(
            prompt_tokens=response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            completion_tokens=response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
            total_tokens=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
        )
        self._record_usage(usage)

        return GenerationResponse(
            content=text,
            model=model_name,
            usage=usage,
            finish_reason="stop",
            is_mock=False,
        )

    async def structured_generate(
        self, request: StructuredGenerationRequest
    ) -> BaseModel:
        schema_json = request.output_schema.model_json_schema()
        enhanced_prompt = (
            f"{request.prompt}\n\n"
            f"Respond ONLY with a valid JSON object conforming to this schema:\n"
            f"{json.dumps(schema_json, indent=2)}"
        )

        gen_request = GenerationRequest(
            prompt=enhanced_prompt,
            system_prompt=request.system_prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        response = await self.generate(gen_request)
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        parsed = json.loads(content)
        return request.output_schema.model_validate(parsed)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        if not self._available:
            yield "[UNAVAILABLE] google-generativeai package not installed."
            return
        client = self._get_client()
        model_name = request.model or self.default_model
        model = client.GenerativeModel(model_name)

        contents = []
        if request.system_prompt:
            contents.append({"role": "user", "parts": [request.system_prompt]})
            contents.append({"role": "model", "parts": ["Understood."]})
        contents.append({"role": "user", "parts": [request.prompt]})

        response = await model.generate_content_async(
            contents,
            generation_config={
                "max_output_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text