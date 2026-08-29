"""MathModel AI — Routing policy.

Determines which model tier to use based on task profile.
Phase 2: Uses TaskProfile dimensions for transparent,
explainable routing decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from mathmodel.config import ModelTier, get_settings
from mathmodel.routing.profile import ComplexityTier, TaskProfile, TaskType


@dataclass
class RoutingExplanation:
    """Explains why a particular tier was selected for a task."""

    selected_tier: ModelTier
    primary_reasons: list[str] = field(default_factory=list)
    triggered_rules: list[str] = field(default_factory=list)
    escalation_reason: Optional[str] = None
    task_profile_snapshot: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "selected_tier": self.selected_tier.value,
            "primary_reasons": self.primary_reasons,
            "triggered_rules": self.triggered_rules,
            "escalation_reason": self.escalation_reason,
            "task_profile_snapshot": self.task_profile_snapshot,
        }


class RoutingPolicy:
    """Policy for selecting model tiers based on task profiles.

    Phase 2: Considers reasoning_requirement, math_requirement,
    review_requirement, long_context_requirement, blast_radius,
    and deadline_pressure in addition to complexity and task_type.
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

    # Dimension escalation rules: which dimensions can push tier up
    DIMENSION_RULES: dict[str, dict[ComplexityTier, ModelTier]] = {
        "reasoning_requirement": {
            ComplexityTier.HIGH: ModelTier.FLAGSHIP_HIGH,
            ComplexityTier.CRITICAL: ModelTier.FLAGSHIP_XHIGH,
        },
        "math_requirement": {
            ComplexityTier.HIGH: ModelTier.FLAGSHIP_HIGH,
            ComplexityTier.CRITICAL: ModelTier.FLAGSHIP_XHIGH,
        },
        "review_requirement": {
            ComplexityTier.HIGH: ModelTier.FLAGSHIP_HIGH,
            ComplexityTier.CRITICAL: ModelTier.FLAGSHIP_XHIGH,
        },
        "long_context_requirement": {
            ComplexityTier.HIGH: ModelTier.FLAGSHIP_HIGH,
            ComplexityTier.CRITICAL: ModelTier.FLAGSHIP_XHIGH,
        },
        "blast_radius": {
            ComplexityTier.HIGH: ModelTier.FLAGSHIP_HIGH,
            ComplexityTier.CRITICAL: ModelTier.FLAGSHIP_XHIGH,
        },
        "coding_requirement": {
            ComplexityTier.CRITICAL: ModelTier.FLAGSHIP_HIGH,
        },
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

    def select_tier(
        self, profile: TaskProfile
    ) -> tuple[ModelTier, RoutingExplanation]:
        """Select the appropriate model tier for a task profile.

        Returns both the tier and a RoutingExplanation.
        """
        reasons: list[str] = []
        rules: list[str] = []

        # Base tier from complexity
        complexity_tier = self.COMPLEXITY_TIER_MAP.get(
            profile.complexity, ModelTier.BALANCED
        )
        reasons.append(f"complexity={profile.complexity.value} -> {complexity_tier.value}")

        # Minimum tier for task type
        minimum_tier = self.get_minimum_tier(profile.task_type)
        if minimum_tier != ModelTier.FAST:
            reasons.append(f"task_type={profile.task_type.value} min={minimum_tier.value}")
            rules.append(f"RULE_TASK_MIN_{minimum_tier.value.upper()}")

        # Take the higher of complexity and task minimum
        base_tier = self._max_tier(complexity_tier, minimum_tier)

        # Apply dimension escalation rules
        dimension_tier = self._apply_dimension_rules(profile, reasons, rules)
        base_tier = self._max_tier(base_tier, dimension_tier)

        # Escalate based on retry count
        escalation_reason = None
        if profile.retry_count > 0:
            escalation_reason = f"retry_count={profile.retry_count}"
            rules.append(f"RULE_RETRY_{profile.retry_count}")

        escalated_tier = self._escalate(base_tier, profile.retry_count)

        if escalated_tier != base_tier:
            reasons.append(f"escalated from {base_tier.value} due to retries")

        # Build explanation
        explanation = RoutingExplanation(
            selected_tier=escalated_tier,
            primary_reasons=reasons,
            triggered_rules=rules,
            escalation_reason=escalation_reason,
            task_profile_snapshot={
                "task_type": profile.task_type.value,
                "complexity": profile.complexity.value,
                "reasoning": profile.reasoning_requirement.value,
                "math": profile.math_requirement.value,
                "coding": profile.coding_requirement.value,
                "review": profile.review_requirement.value,
                "long_context": profile.long_context_requirement.value,
                "blast_radius": profile.blast_radius.value,
                "retry_count": profile.retry_count,
            },
        )

        return escalated_tier, explanation

    def _apply_dimension_rules(
        self,
        profile: TaskProfile,
        reasons: list[str],
        rules: list[str],
    ) -> ModelTier:
        """Apply dimension-based escalation rules.

        Each dimension can independently push the tier up.
        """
        max_dimension_tier = ModelTier.FAST

        dimension_values = {
            "reasoning_requirement": profile.reasoning_requirement,
            "math_requirement": profile.math_requirement,
            "review_requirement": profile.review_requirement,
            "long_context_requirement": profile.long_context_requirement,
            "blast_radius": profile.blast_radius,
            "coding_requirement": profile.coding_requirement,
        }

        for dim_name, dim_value in dimension_values.items():
            if dim_name in self.DIMENSION_RULES:
                rules_for_dim = self.DIMENSION_RULES[dim_name]
                if dim_value in rules_for_dim:
                    tier = rules_for_dim[dim_value]
                    if self._tier_order(tier) > self._tier_order(max_dimension_tier):
                        max_dimension_tier = tier
                        reasons.append(
                            f"{dim_name}={dim_value.value} -> {tier.value}"
                        )
                        rules.append(
                            f"RULE_{dim_name.upper()}_{dim_value.value.upper()}"
                        )

        return max_dimension_tier

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
        return a if RoutingPolicy._tier_order(a) >= RoutingPolicy._tier_order(b) else b

    @staticmethod
    def _tier_order(tier: ModelTier) -> int:
        """Return numeric order for a tier."""
        order = {
            ModelTier.FAST: 0,
            ModelTier.BALANCED: 1,
            ModelTier.FLAGSHIP_HIGH: 2,
            ModelTier.FLAGSHIP_XHIGH: 3,
            ModelTier.FLAGSHIP_MAX: 4,
        }
        return order.get(tier, 0)