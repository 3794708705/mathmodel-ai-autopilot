"""Tests for model routing.

Phase 1 + Phase 2: tests the RoutingPolicy and ModelRouter.
Updated for Phase 2 tuple return from select_tier.
"""

import asyncio

import pytest

from mathmodel.config import ModelTier
from mathmodel.routing.policy import RoutingPolicy
from mathmodel.routing.profile import ComplexityTier, TaskProfile, TaskType
from mathmodel.routing.router import ModelRouter


class TestTaskProfile:
    """Test TaskProfile dataclass and factory."""

    def test_default_profile(self):
        profile = TaskProfile(task_type=TaskType.CRUD)
        assert profile.task_type == TaskType.CRUD
        assert profile.complexity == ComplexityTier.MEDIUM
        assert profile.retry_count == 0

    def test_for_task_type_math_modeling(self):
        profile = TaskProfile.for_task_type(TaskType.MATHEMATICAL_MODELING)
        assert profile.complexity == ComplexityTier.CRITICAL
        assert profile.reasoning_requirement == ComplexityTier.CRITICAL
        assert profile.math_requirement == ComplexityTier.CRITICAL
        assert profile.blast_radius == ComplexityTier.CRITICAL

    def test_for_task_type_documentation(self):
        profile = TaskProfile.for_task_type(TaskType.DOCUMENTATION)
        assert profile.complexity == ComplexityTier.LOW
        assert profile.reasoning_requirement == ComplexityTier.LOW

    def test_for_task_type_sandbox_security(self):
        profile = TaskProfile.for_task_type(TaskType.SANDBOX_SECURITY)
        assert profile.security_risk == ComplexityTier.CRITICAL
        assert profile.blast_radius == ComplexityTier.CRITICAL

    def test_for_task_type_with_overrides(self):
        profile = TaskProfile.for_task_type(
            TaskType.CODE_GENERATION,
            complexity=ComplexityTier.CRITICAL,
            retry_count=2,
        )
        assert profile.complexity == ComplexityTier.CRITICAL
        assert profile.retry_count == 2
        assert profile.coding_requirement == ComplexityTier.HIGH

    def test_all_task_types_exist(self):
        """Test that all expected task types are defined."""
        task_types = list(TaskType)
        assert TaskType.MATHEMATICAL_MODELING in task_types
        assert TaskType.PROBLEM_UNDERSTANDING in task_types
        assert TaskType.DATA_ANALYSIS in task_types
        assert TaskType.CODE_GENERATION in task_types
        assert TaskType.VALIDATION in task_types
        assert TaskType.PAPER_GENERATION in task_types
        assert TaskType.DOCUMENTATION in task_types


class TestRoutingPolicy:
    """Test RoutingPolicy logic — Phase 2 tuple return."""

    def test_select_tier_by_complexity(self):
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.LOW,
        )
        tier, _ = policy.select_tier(profile)
        assert tier == ModelTier.FAST

    def test_select_tier_medium(self):
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
        )
        tier, _ = policy.select_tier(profile)
        assert tier == ModelTier.BALANCED

    def test_select_tier_critical(self):
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.CRITICAL,
        )
        tier, _ = policy.select_tier(profile)
        assert tier == ModelTier.FLAGSHIP_XHIGH

    def test_minimum_tier_enforcement(self):
        """Test that minimum tier for task type is enforced."""
        policy = RoutingPolicy()
        # Mathematical modeling has minimum FLAGSHIP_HIGH, but
        # the reasoning_requirement=CRITICAL also pushes tier up
        profile = TaskProfile.for_task_type(
            TaskType.MATHEMATICAL_MODELING,
            complexity=ComplexityTier.LOW,
        )
        tier, _ = policy.select_tier(profile)
        # At minimum FLAGSHIP_HIGH, but dimension rules may push higher
        assert policy._tier_order(tier) >= policy._tier_order(ModelTier.FLAGSHIP_HIGH)

    def test_escalation_chain(self):
        policy = RoutingPolicy()
        assert policy.escalate(ModelTier.FAST) == ModelTier.BALANCED
        assert policy.escalate(ModelTier.BALANCED) == ModelTier.FLAGSHIP_HIGH
        assert policy.escalate(ModelTier.FLAGSHIP_HIGH) == ModelTier.FLAGSHIP_XHIGH
        assert policy.escalate(ModelTier.FLAGSHIP_XHIGH) == ModelTier.FLAGSHIP_MAX
        assert policy.escalate(ModelTier.FLAGSHIP_MAX) is None

    def test_retry_escalation(self):
        """Test that retries escalate the model tier."""
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            retry_count=2,
        )
        tier, _ = policy.select_tier(profile)
        # Medium -> BALANCED, then 2 escalations -> FLAGSHIP_XHIGH
        assert tier == ModelTier.FLAGSHIP_XHIGH

    def test_retry_escalation_capped(self):
        """Test that escalation doesn't exceed FLAGSHIP_MAX."""
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            retry_count=10,
        )
        tier, _ = policy.select_tier(profile)
        assert tier == ModelTier.FLAGSHIP_MAX

    def test_documentation_minimum(self):
        """Test that documentation has minimum FAST."""
        policy = RoutingPolicy()
        profile = TaskProfile.for_task_type(
            TaskType.DOCUMENTATION,
            complexity=ComplexityTier.TRIVIAL,
        )
        tier, _ = policy.select_tier(profile)
        assert tier == ModelTier.FAST


class TestModelRouter:
    """Test ModelRouter — Phase 2 tuple return."""

    def test_select_tier(self):
        router = ModelRouter()
        profile = TaskProfile.for_task_type(TaskType.DOCUMENTATION)
        tier, explanation = router.select_tier(profile)
        assert tier == ModelTier.FAST
        assert explanation is not None

    def test_select_tier_math(self):
        router = ModelRouter()
        profile = TaskProfile.for_task_type(TaskType.MATHEMATICAL_MODELING)
        tier, _ = router.select_tier(profile)
        assert tier in (ModelTier.FLAGSHIP_HIGH, ModelTier.FLAGSHIP_XHIGH)

    def test_get_provider_for_tier(self):
        router = ModelRouter()
        provider = router.get_provider_for_tier(ModelTier.FAST)
        assert provider is not None
        assert provider.provider_name in ("mock", "openai", "google", "anthropic")

    def test_route_generate(self):
        """Test route_generate with MockProvider."""
        router = ModelRouter()
        profile = TaskProfile.for_task_type(TaskType.DOCUMENTATION)

        async def _run():
            return await router.route_generate(
                profile=profile,
                prompt="Test prompt",
            )

        response = asyncio.run(_run())
        assert response is not None
        assert response.content is not None
        assert response.is_mock is True