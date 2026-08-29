"""MathModel AI — Provider registry.

Central registry for model provider instances.
"""

from __future__ import annotations

import logging
from typing import Optional

from mathmodel.config import ProviderType, get_settings
from mathmodel.providers.anthropic import AnthropicProvider
from mathmodel.providers.base import BaseModelProvider
from mathmodel.providers.google import GoogleProvider
from mathmodel.providers.mock import MockProvider
from mathmodel.providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)


class ProviderUnavailableError(Exception):
    """Raised when a real provider is required but no credential is available."""


class ProviderRegistry:
    """Registry of available model providers.

    Lazily instantiates providers on first access.
    Default (development) mode falls back to MockProvider when a real
    provider's API key is missing — but strict mode raises
    ProviderUnavailableError instead. Strict mode is REQUIRED when
    RealityContext.mock_allowed = False.
    """

    def __init__(self):
        self._providers: dict[tuple[ProviderType, bool], BaseModelProvider] = {}
        self._settings = get_settings()

    def get_provider(
        self,
        provider_type: Optional[ProviderType] = None,
        strict: bool = False,
    ) -> BaseModelProvider:
        """Get a provider instance by type. Defaults to the configured default.

        strict=True: never fall back to Mock — raise ProviderUnavailableError
        when the real provider's credential is missing.
        """
        if provider_type is None:
            provider_type = self._settings.default_provider

        cache_key = (provider_type, strict)
        if cache_key not in self._providers:
            self._providers[cache_key] = self._create_provider(provider_type, strict=strict)

        return self._providers[cache_key]

    def _create_provider(
        self, provider_type: ProviderType, strict: bool = False,
    ) -> BaseModelProvider:
        """Create a new provider instance.

        Development mode (strict=False): falls back to MockProvider when
        a real provider's API key is not configured.
        Strict mode (strict=True): raises ProviderUnavailableError.
        """
        settings = self._settings

        if provider_type == ProviderType.MOCK:
            if strict:
                raise ProviderUnavailableError(
                    "Mock provider requested in strict mode "
                    "(mock_allowed=false) — no Mock fallback"
                )
            logger.info("Using MockProvider (explicitly configured)")
            return MockProvider(default_model="mock-model")

        def _mock_or_raise(
            provider_label: str, env_var: str, default_model: str,
        ) -> MockProvider:
            if strict:
                raise ProviderUnavailableError(
                    f"{provider_label} unavailable: {env_var} not configured "
                    "(strict mode — no Mock fallback)"
                )
            logger.warning(
                "%s not configured. Falling back to MockProvider. "
                "All %s requests will return is_mock=True.",
                env_var, provider_label,
            )
            return MockProvider(
                default_model=default_model,
                fixed_response=f"[MOCK] {provider_label} API key not configured.",
            )

        if provider_type == ProviderType.OPENAI:
            api_key = settings.get_api_key(ProviderType.OPENAI)
            if not api_key:
                return _mock_or_raise(
                    "OpenAI", "OPENAI_API_KEY", settings.openai_default_model,
                )
            return OpenAIProvider(
                api_key=api_key,
                default_model=settings.openai_default_model,
            )

        if provider_type == ProviderType.GOOGLE:
            api_key = settings.get_api_key(ProviderType.GOOGLE)
            if not api_key:
                return _mock_or_raise(
                    "Google", "GOOGLE_API_KEY", settings.google_default_model,
                )
            return GoogleProvider(
                api_key=api_key,
                default_model=settings.google_default_model,
            )

        if provider_type == ProviderType.ANTHROPIC:
            api_key = settings.get_api_key(ProviderType.ANTHROPIC)
            if not api_key:
                return _mock_or_raise(
                    "Anthropic", "ANTHROPIC_API_KEY", settings.anthropic_default_model,
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