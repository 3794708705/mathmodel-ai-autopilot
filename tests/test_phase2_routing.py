"""Tests for Phase 2 upgraded RoutingPolicy (TD-1).

Covers: dimension-based routing, RoutingExplanation,
escalation behavior, and minimum tier enforcement.
"""

import pytest

from mathmodel.config import ModelTier
from mathmodel.routing.policy import RoutingExplanation, RoutingPolicy
from mathmodel.routing.profile import ComplexityTier, TaskProfile, TaskType


class TestRoutingExplanation:
    def test_explanation_created(self):
        policy = RoutingPolicy()
        profile = TaskProfile.for_task_type(TaskType.MATHEMATICAL_MODELING)
        tier, explanation = policy.select_tier(profile)

        assert isinstance(explanation, RoutingExplanation)
        assert explanation.selected_tier == tier
        assert len(explanation.primary_reasons) > 0
        assert "task_type" in explanation.task_profile_snapshot

    def test_explanation_as_dict(self):
        policy = RoutingPolicy()
        profile = TaskProfile.for_task_type(TaskType.CRUD)
        _, explanation = policy.select_tier(profile)

        d = explanation.as_dict()
        assert "selected_tier" in d
        assert "primary_reasons" in d
        assert "triggered_rules" in d
        assert "task_profile_snapshot" in d


class TestDimensionBasedRouting:
    """TD-1: TaskProfile dimensions must affect routing tier."""

    def test_math_requirement_affects_tier(self):
        policy = RoutingPolicy()
        profile_low = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            math_requirement=ComplexityTier.LOW,
        )
        profile_high = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            math_requirement=ComplexityTier.CRITICAL,
        )

        tier_low, _ = policy.select_tier(profile_low)
        tier_high, _ = policy.select_tier(profile_high)

        # High math requirement should push tier up
        assert policy._tier_order(tier_high) >= policy._tier_order(tier_low)

    def test_reasoning_requirement_affects_tier(self):
        policy = RoutingPolicy()
        profile_low = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            reasoning_requirement=ComplexityTier.LOW,
        )
        profile_high = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            reasoning_requirement=ComplexityTier.CRITICAL,
        )

        tier_low, _ = policy.select_tier(profile_low)
        tier_high, _ = policy.select_tier(profile_high)

        assert policy._tier_order(tier_high) >= policy._tier_order(tier_low)

    def test_blast_radius_cannot_route_too_low(self):
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.LOW,
            blast_radius=ComplexityTier.CRITICAL,
        )
        tier, explanation = policy.select_tier(profile)

        # High blast radius should elevate tier
        assert tier != ModelTier.FAST
        assert any("blast_radius" in r for r in explanation.primary_reasons)

    def test_retry_escalation(self):
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            retry_count=3,
        )
        tier, explanation = policy.select_tier(profile)

        # Should be escalated above BALANCED
        assert policy._tier_order(tier) > policy._tier_order(ModelTier.BALANCED)
        assert explanation.escalation_reason is not None

    def test_minimum_tier_respected(self):
        policy = RoutingPolicy()
        profile = TaskProfile.for_task_type(
            TaskType.MATHEMATICAL_MODELING,
            complexity=ComplexityTier.LOW,
        )
        tier, _ = policy.select_tier(profile)

        # Mathematical modeling has minimum FLAGSHIP_HIGH
        assert policy._tier_order(tier) >= policy._tier_order(ModelTier.FLAGSHIP_HIGH)

    def test_maximum_tier_bounded(self):
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.CRITICAL,
            math_requirement=ComplexityTier.CRITICAL,
            reasoning_requirement=ComplexityTier.CRITICAL,
            blast_radius=ComplexityTier.CRITICAL,
            retry_count=10,
        )
        tier, _ = policy.select_tier(profile)

        # Cannot exceed FLAGSHIP_MAX
        assert tier == ModelTier.FLAGSHIP_MAX

    def test_routing_explanation_generated(self):
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.MODEL_EXPLORATION,
            complexity=ComplexityTier.HIGH,
            math_requirement=ComplexityTier.HIGH,
            reasoning_requirement=ComplexityTier.HIGH,
        )
        _, explanation = policy.select_tier(profile)

        # Must have reasons
        assert len(explanation.primary_reasons) >= 2
        assert len(explanation.triggered_rules) >= 1

    def test_documentation_stays_low(self):
        policy = RoutingPolicy()
        profile = TaskProfile.for_task_type(
            TaskType.DOCUMENTATION,
            complexity=ComplexityTier.TRIVIAL,
        )
        tier, _ = policy.select_tier(profile)

        assert tier == ModelTier.FAST


class TestEscalationChain:
    def test_full_chain(self):
        policy = RoutingPolicy()
        assert policy.escalate(ModelTier.FAST) == ModelTier.BALANCED
        assert policy.escalate(ModelTier.BALANCED) == ModelTier.FLAGSHIP_HIGH
        assert policy.escalate(ModelTier.FLAGSHIP_HIGH) == ModelTier.FLAGSHIP_XHIGH
        assert policy.escalate(ModelTier.FLAGSHIP_XHIGH) == ModelTier.FLAGSHIP_MAX
        assert policy.escalate(ModelTier.FLAGSHIP_MAX) is None

    def test_escalation_capped(self):
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            retry_count=100,
        )
        tier, _ = policy.select_tier(profile)
        assert tier == ModelTier.FLAGSHIP_MAX


class TestBackwardCompatibility:
    """Ensure Phase 1 tests still work with the new RoutingPolicy signature."""

    def test_select_tier_returns_tuple(self):
        policy = RoutingPolicy()
        profile = TaskProfile.for_task_type(TaskType.CRUD)
        result = policy.select_tier(profile)
        assert isinstance(result, tuple)
        assert len(result) == 2
        tier, explanation = result
        assert isinstance(tier, ModelTier)
        assert isinstance(explanation, RoutingExplanation)

    def test_basic_routing_unchanged(self):
        """Basic routing behavior should not regress."""
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
        )
        tier, _ = policy.select_tier(profile)
        assert tier == ModelTier.BALANCED