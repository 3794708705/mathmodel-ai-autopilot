"""MathModel AI — Model router.

Orchestrates model selection: profiles the task, applies routing policy,
and returns the appropriate provider for the selected tier.
"""

from __future__ import annotations

import logging
from typing import Optional

from mathmodel.config import ModelTier, ProviderType
from mathmodel.providers.base import (
    BaseModelProvider,
    GenerationRequest,
    GenerationResponse,
    StructuredGenerationRequest,
)
from mathmodel.providers.registry import get_provider_registry
from mathmodel.routing.policy import RoutingPolicy
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

    def select_tier(self, profile: TaskProfile) -> ModelTier:
        """Select the appropriate model tier for a task profile."""
        return self._policy.select_tier(profile)

    def get_provider_for_tier(
        self, tier: ModelTier, preferred_provider: Optional[ProviderType] = None
    ) -> BaseModelProvider:
        """Get a provider instance appropriate for the given tier."""
        provider_type = preferred_provider or self._get_default_provider_for_tier(tier)
        return self._registry.get_provider(provider_type)

    async def route_generate(
        self,
        profile: TaskProfile,
        prompt: str,
        system_prompt: Optional[str] = None,
        preferred_provider: Optional[ProviderType] = None,
        **kwargs,
    ) -> GenerationResponse:
        """Route a generation request through the model router.

        Profiles the task, selects the tier, and delegates to the
        appropriate provider.
        """
        tier = self.select_tier(profile)
        provider = self.get_provider_for_tier(tier, preferred_provider)

        logger.info(
            "Routing task type=%s tier=%s provider=%s retry=%d",
            profile.task_type.value,
            tier.value,
            provider.provider_name,
            profile.retry_count,
        )

        request = GenerationRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

        return await provider.generate(request)

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
        tier = self.select_tier(profile)
        provider = self.get_provider_for_tier(tier, preferred_provider)

        logger.info(
            "Routing structured task type=%s tier=%s provider=%s",
            profile.task_type.value,
            tier.value,
            provider.provider_name,
        )

        request = StructuredGenerationRequest(
            prompt=prompt,
            output_schema=output_schema,
            system_prompt=system_prompt,
            **kwargs,
        )

        return await provider.structured_generate(request)

    @staticmethod
    def _get_default_provider_for_tier(tier: ModelTier) -> ProviderType:
        """Map model tier to default provider type."""
        # In production, this would be more sophisticated
        # For now, OpenAI is the default for all tiers
        return ProviderType.OPENAI