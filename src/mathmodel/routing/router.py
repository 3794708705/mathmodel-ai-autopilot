"""MathModel AI — Model router.

Orchestrates model selection: profiles the task, applies routing policy,
and returns the appropriate provider for the selected tier.
"""

from __future__ import annotations

import logging
from typing import Optional

from mathmodel.config import ModelTier, ProviderType, get_settings
from mathmodel.providers.base import (
    BaseModelProvider,
    GenerationRequest,
    GenerationResponse,
    StructuredGenerationRequest,
)
from mathmodel.providers.registry import get_provider_registry
from mathmodel.routing.policy import RoutingExplanation, RoutingPolicy
from mathmodel.routing.profile import TaskProfile

logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes tasks to the appropriate model provider and tier.

    Execution chain:
        Task -> TaskProfiler -> RoutingPolicy -> ModelRouter -> ModelProvider -> Selected Model
    """

    def __init__(self):
        self._policy = RoutingPolicy()
        self._registry = get_provider_registry()
        self._settings = get_settings()

    def select_tier(
        self, profile: TaskProfile
    ) -> tuple[ModelTier, RoutingExplanation]:
        """Select the appropriate model tier for a task profile.

        Returns both the tier and an explanation of why it was chosen.
        """
        return self._policy.select_tier(profile)

    def get_provider_for_tier(
        self,
        tier: ModelTier,
        preferred_provider: Optional[ProviderType] = None,
        strict: bool = False,
    ) -> BaseModelProvider:
        """Get a provider instance appropriate for the given tier.

        strict=True: raise ProviderUnavailableError instead of falling
        back to Mock when a real provider's credential is missing.
        Required when RealityContext.mock_allowed = False.
        """
        provider_type = preferred_provider or self._get_default_provider_for_tier(tier)
        provider = self._registry.get_provider(provider_type, strict=strict)

        if provider.provider_name == "mock":
            logger.warning(
                "Provider %s fell back to MockProvider (API key missing or unavailable). "
                "Responses will have is_mock=True.",
                provider_type.value,
            )

        return provider

    async def route_generate(
        self,
        profile: TaskProfile,
        prompt: str,
        system_prompt: Optional[str] = None,
        preferred_provider: Optional[ProviderType] = None,
        strict: bool = False,
        **kwargs,
    ) -> GenerationResponse:
        """Route a generation request through the model router.

        Profiles the task, selects the tier, and delegates to the
        appropriate provider.
        """
        tier, explanation = self.select_tier(profile)
        provider = self.get_provider_for_tier(tier, preferred_provider, strict=strict)

        logger.info(
            "Routing task type=%s tier=%s provider=%s retry=%d reasons=%s",
            profile.task_type.value,
            tier.value,
            provider.provider_name,
            profile.retry_count,
            explanation.primary_reasons,
        )

        request = GenerationRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

        response = await provider.generate(request)

        if response.is_mock:
            logger.warning(
                "Task type=%s received mock response from provider=%s. "
                "Real LLM was not used.",
                profile.task_type.value,
                provider.provider_name,
            )

        return response

    async def route_structured_generate(
        self,
        profile: TaskProfile,
        prompt: str,
        output_schema,
        system_prompt: Optional[str] = None,
        preferred_provider: Optional[ProviderType] = None,
        **kwargs,
    ):
        """Route a structured generation request through the model router."""
        tier, explanation = self.select_tier(profile)
        provider = self.get_provider_for_tier(tier, preferred_provider)

        logger.info(
            "Routing structured task type=%s tier=%s provider=%s reasons=%s",
            profile.task_type.value,
            tier.value,
            provider.provider_name,
            explanation.primary_reasons,
        )

        # Model selection by tier: reasoning-heavy tasks get the stronger
        # model when the provider supports multiple models. Slug comes
        # from configuration, never hardcoded per task.
        model_override = self._select_model_for_tier(provider, tier)
        if model_override and "model" not in kwargs:
            kwargs["model"] = model_override

        request = StructuredGenerationRequest(
            prompt=prompt,
            output_schema=output_schema,
            system_prompt=system_prompt,
            **kwargs,
        )

        return await provider.structured_generate(request)

    def _select_model_for_tier(
        self, provider: BaseModelProvider, tier: ModelTier
    ) -> Optional[str]:
        """Choose a configured model slug for the tier, if applicable."""
        settings = get_settings()
        high_tiers = (
            ModelTier.FLAGSHIP_HIGH,
            ModelTier.FLAGSHIP_XHIGH,
            ModelTier.FLAGSHIP_MAX,
        )
        if tier in high_tiers and provider.provider_name == "deepseek":
            return settings.deepseek_reasoning_model
        return None

    @staticmethod
    def _get_default_provider_for_tier(tier: ModelTier) -> ProviderType:
        """Map model tier to default provider type.

        Phase 1: all tiers default to the configured default_provider.
        Phase 2+: this will map tiers to specific providers/models.
        """
        settings = get_settings()
        return settings.default_provider