"""Tests for model providers."""

import pytest
from pydantic import BaseModel

from mathmodel.providers.base import (
    GenerationRequest,
    GenerationResponse,
    StructuredGenerationRequest,
    UsageInfo,
)
from mathmodel.providers.mock import MockProvider
from mathmodel.providers.registry import ProviderRegistry, get_provider_registry


class TestUsageInfo:
    """Test UsageInfo dataclass."""

    def test_default_usage(self):
        usage = UsageInfo()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost_usd == 0.0

    def test_usage_with_values(self):
        usage = UsageInfo(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cost_usd == 0.01


class TestGenerationRequest:
    """Test GenerationRequest dataclass."""

    def test_default_request(self):
        req = GenerationRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.system_prompt is None
        assert req.model is None
        assert req.max_tokens == 4096
        assert req.temperature == 0.7

    def test_request_with_options(self):
        req = GenerationRequest(
            prompt="Hello",
            system_prompt="You are helpful.",
            model="gpt-4o",
            max_tokens=1000,
            temperature=0.3,
            stop_sequences=["END"],
        )
        assert req.system_prompt == "You are helpful."
        assert req.model == "gpt-4o"
        assert req.max_tokens == 1000
        assert req.temperature == 0.3
        assert req.stop_sequences == ["END"]


class TestGenerationResponse:
    """Test GenerationResponse dataclass."""

    def test_default_response(self):
        resp = GenerationResponse(content="Hello", model="test-model")
        assert resp.content == "Hello"
        assert resp.model == "test-model"
        assert resp.finish_reason == "stop"
        assert resp.is_mock is False

    def test_mock_response(self):
        resp = GenerationResponse(
            content="Mock response",
            model="mock",
            is_mock=True,
        )
        assert resp.is_mock is True


class TestMockProvider:
    """Test MockProvider implementation."""

    def test_generate_sync(self):
        """Test MockProvider.generate synchronously."""
        import asyncio
        provider = MockProvider()
        request = GenerationRequest(prompt="What is 2+2?")
        response = asyncio.run(provider.generate(request))

        assert response.content is not None
        assert "[MOCK RESPONSE]" in response.content
        assert response.is_mock is True
        assert response.model == "mock-model"

    def test_generate_with_fixed_response(self):
        import asyncio
        provider = MockProvider(fixed_response="Fixed answer")
        request = GenerationRequest(prompt="Any prompt")
        response = asyncio.run(provider.generate(request))

        assert response.content == "Fixed answer"
        assert response.is_mock is True

    def test_generate_with_custom_model(self):
        import asyncio
        provider = MockProvider(default_model="custom-mock")
        request = GenerationRequest(prompt="Test", model="other-model")
        response = asyncio.run(provider.generate(request))

        assert response.model == "other-model"

    def test_structured_generate(self):
        import asyncio

        class TestSchema(BaseModel):
            name: str = "default"
            value: int = 0

        provider = MockProvider()
        request = StructuredGenerationRequest(
            prompt="Generate test data",
            output_schema=TestSchema,
        )
        result = asyncio.run(provider.structured_generate(request))

        assert isinstance(result, TestSchema)
        assert result.name == "default"
        assert result.value == 0

    def test_get_usage(self):
        import asyncio
        provider = MockProvider()
        request = GenerationRequest(prompt="Test prompt with several words")
        asyncio.run(provider.generate(request))

        usage = provider.get_usage()
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens > 0

    def test_provider_name(self):
        provider = MockProvider()
        assert provider.provider_name == "mock"

    def test_stream(self):
        import asyncio

        async def collect():
            provider = MockProvider(fixed_response="one two three")
            request = GenerationRequest(prompt="Count")
            words = []
            async for word in provider.stream(request):
                words.append(word)
            return words

        words = asyncio.run(collect())
        assert len(words) == 3
        assert "one" in words[0]


class TestProviderRegistry:
    """Test ProviderRegistry."""

    def test_get_mock_provider(self):
        registry = ProviderRegistry()
        provider = registry.get_provider()
        assert provider.provider_name == "mock"

    def test_get_provider_caching(self):
        registry = ProviderRegistry()
        p1 = registry.get_provider()
        p2 = registry.get_provider()
        assert p1 is p2

    def test_reset(self):
        registry = ProviderRegistry()
        p1 = registry.get_provider()
        registry.reset()
        p2 = registry.get_provider()
        assert p1 is not p2

    def test_get_provider_without_api_key(self):
        """Test that providers without API keys fall back to Mock."""
        from mathmodel.config import ProviderType
        registry = ProviderRegistry()
        provider = registry.get_provider(ProviderType.OPENAI)
        assert provider.provider_name == "mock"