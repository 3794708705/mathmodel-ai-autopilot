"""MathModel AI — Routing policy.

Determines which model tier to use based on task profile.
"""

from __future__ import annotations

import json
from typing import Optional

from mathmodel.config import ModelTier, get_settings
from mathmodel.routing.profile import ComplexityTier, TaskProfile, TaskType


class RoutingPolicy:
    """Policy for selecting model tiers based on task profiles.

    Implements the escalation chain and minimum tier enforcement.
    """

    # Escalation chain: ordered from lowest to highest capability
    ESCALATION_CHAIN: list[ModelTier] = [
        ModelTier.FAST,
        ModelTier.BALANCED,
        ModelTier.FLAGSHIP_HIGH,
        ModelTier.FLAGSHIP_XHIGH,
        ModelTier.FLAGSHIP_MAX,
    ]

    # Default tier mapping based on complexity
    COMPLEXITY_TIER_MAP: dict[ComplexityTier, ModelTier] = {
        ComplexityTier.TRIVIAL: ModelTier.FAST,
        ComplexityTier.LOW: ModelTier.FAST,
        ComplexityTier.MEDIUM: ModelTier.BALANCED,
        ComplexityTier.HIGH: ModelTier.FLAGSHIP_HIGH,
        ComplexityTier.CRITICAL: ModelTier.FLAGSHIP_XHIGH,
    }

    def __init__(self):
        self._settings = get_settings()
        self._minimum_tiers = self._load_minimum_tiers()

    def _load_minimum_tiers(self) -> dict[str, ModelTier]:
        """Load minimum tier requirements from settings."""
        try:
            raw = json.loads(self._settings.minimum_model_tiers)
            return {
                task_type: ModelTier(tier)
                for task_type, tier in raw.items()
            }
        except (json.JSONDecodeError, ValueError):
            return {}

    def get_minimum_tier(self, task_type: TaskType) -> ModelTier:
        """Get the minimum allowed tier for a task type."""
        return self._minimum_tiers.get(task_type.value, ModelTier.FAST)

    def select_tier(self, profile: TaskProfile) -> ModelTier:
        """Select the appropriate model tier for a task profile.

        The selected tier is the maximum of:
        1. The tier implied by the task's complexity
        2. The minimum tier for the task type
        3. The escalated tier based on retry count
        """
        # Base tier from complexity
        complexity_tier = self.COMPLEXITY_TIER_MAP.get(
            profile.complexity, ModelTier.BALANCED
        )

        # Minimum tier for task type
        minimum_tier = self.get_minimum_tier(profile.task_type)

        # Take the higher of the two
        base_tier = self._max_tier(complexity_tier, minimum_tier)

        # Escalate based on retry count
        escalated_tier = self._escalate(base_tier, profile.retry_count)

        return escalated_tier

    def escalate(self, current_tier: ModelTier) -> Optional[ModelTier]:
        """Move one step up the escalation chain.

        Returns None if already at max.
        """
        try:
            idx = self.ESCALATION_CHAIN.index(current_tier)
            if idx < len(self.ESCALATION_CHAIN) - 1:
                return self.ESCALATION_CHAIN[idx + 1]
            return None
        except ValueError:
            return ModelTier.BALANCED

    def _escalate(self, base_tier: ModelTier, retry_count: int) -> ModelTier:
        """Escalate tier based on retry count."""
        current = base_tier
        for _ in range(retry_count):
            next_tier = self.escalate(current)
            if next_tier is None:
                break
            current = next_tier
        return current

    @staticmethod
    def _max_tier(a: ModelTier, b: ModelTier) -> ModelTier:
        """Return the higher-capability tier."""
        order = {
            ModelTier.FAST: 0,
            ModelTier.BALANCED: 1,
            ModelTier.FLAGSHIP_HIGH: 2,
            ModelTier.FLAGSHIP_XHIGH: 3,
            ModelTier.FLAGSHIP_MAX: 4,
        }
        return a if order.get(a, 0) >= order.get(b, 0) else b