"""MathModel AI — Provider registry.

Central registry for model provider instances.
"""

from __future__ import annotations

from typing import Optional

from mathmodel.config import ProviderType, get_settings
from mathmodel.providers.anthropic import AnthropicProvider
from mathmodel.providers.base import BaseModelProvider
from mathmodel.providers.google import GoogleProvider
from mathmodel.providers.mock import MockProvider
from mathmodel.providers.openai import OpenAIProvider


class ProviderRegistry:
    """Registry of available model providers.

    Lazily instantiates providers on first access.
    """

    def __init__(self):
        self._providers: dict[ProviderType, BaseModelProvider] = {}
        self._settings = get_settings()

    def get_provider(self, provider_type: Optional[ProviderType] = None) -> BaseModelProvider:
        """Get a provider instance by type. Defaults to the configured default."""
        if provider_type is None:
            provider_type = self._settings.default_provider

        if provider_type not in self._providers:
            self._providers[provider_type] = self._create_provider(provider_type)

        return self._providers[provider_type]

    def _create_provider(self, provider_type: ProviderType) -> BaseModelProvider:
        """Create a new provider instance."""
        settings = self._settings

        if provider_type == ProviderType.MOCK:
            return MockProvider(default_model="mock-model")

        if provider_type == ProviderType.OPENAI:
            api_key = settings.get_api_key(ProviderType.OPENAI)
            if not api_key:
                return MockProvider(
                    default_model=settings.openai_default_model,
                    fixed_response="[MOCK] OpenAI API key not configured.",
                )
            return OpenAIProvider(
                api_key=api_key,
                default_model=settings.openai_default_model,
            )

        if provider_type == ProviderType.GOOGLE:
            api_key = settings.get_api_key(ProviderType.GOOGLE)
            if not api_key:
                return MockProvider(
                    default_model=settings.google_default_model,
                    fixed_response="[MOCK] Google API key not configured.",
                )
            return GoogleProvider(
                api_key=api_key,
                default_model=settings.google_default_model,
            )

        if provider_type == ProviderType.ANTHROPIC:
            api_key = settings.get_api_key(ProviderType.ANTHROPIC)
            if not api_key:
                return MockProvider(
                    default_model=settings.anthropic_default_model,
                    fixed_response="[MOCK] Anthropic API key not configured.",
                )
            return AnthropicProvider(
                api_key=api_key,
                default_model=settings.anthropic_default_model,
            )

        raise ValueError(f"Unknown provider type: {provider_type}")

    def reset(self) -> None:
        """Clear all cached providers (useful for testing)."""
        self._providers.clear()


# Global singleton
_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry singleton."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry