"""MathModel AI — Phase 7D: Competition Routing Policy.

Deadline-aware model tier selection. Determines CapabilityTier
from task + runtime mode + criticality + budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from mathmodel.config import ModelTier, ProviderType
from mathmodel.runtime import RuntimeMode, RuntimeAction


# ═══════════════════════════════════════════════════════════════
# Task Criticality
# ═══════════════════════════════════════════════════════════════

class TaskCriticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ═══════════════════════════════════════════════════════════════
# Routing Decision
# ═══════════════════════════════════════════════════════════════

@dataclass
class RoutingDecision:
    routing_decision_id: str = field(default_factory=lambda: f"RD-{uuid4().hex[:8]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str = ""
    task_type: str = ""
    criticality: TaskCriticality = TaskCriticality.MEDIUM
    runtime_mode: RuntimeMode = RuntimeMode.STANDARD
    runtime_decision_id: str = ""  # Bound to RuntimeDecision
    authorization_result: str = ""  # "ALLOW" / "HUMAN_REVIEW" / "BLOCK"
    capability_tier: ModelTier = ModelTier.BALANCED
    provider_type: ProviderType = ProviderType.MOCK
    model_slug: str = ""
    max_attempts: int = 3
    max_output_tokens: int = 4096
    reasoning_effort: str = "medium"
    allow_parallel: bool = False
    parallelism_limit: int = 1
    timeout_seconds: int = 120
    reservation_ids: list[str] = field(default_factory=list)
    rationale_codes: list[str] = field(default_factory=list)
    fallback_policy: str = "none"
    escalation_policy: str = "none"
    policy_version: str = "1.0"


# ═══════════════════════════════════════════════════════════════
# Routing Matrix
# ═══════════════════════════════════════════════════════════════

# Task → default criticality
TASK_CRITICALITY: dict[str, TaskCriticality] = {
    "problem_understanding": TaskCriticality.HIGH,
    "model_exploration": TaskCriticality.MEDIUM,
    "model_jury": TaskCriticality.HIGH,
    "mathematical_modeling": TaskCriticality.HIGH,
    "code_generation": TaskCriticality.HIGH,
    "solver_execution": TaskCriticality.CRITICAL,
    "validation": TaskCriticality.CRITICAL,
    "sensitivity": TaskCriticality.MEDIUM,
    "robustness": TaskCriticality.MEDIUM,
    "red_team": TaskCriticality.HIGH,
    "model_repair": TaskCriticality.CRITICAL,
    "paper_generation": TaskCriticality.MEDIUM,
    "paper_repair": TaskCriticality.MEDIUM,
    "citation_verification": TaskCriticality.MEDIUM,
    "literature_search": TaskCriticality.LOW,
    "format_fix": TaskCriticality.LOW,
    "submission_check": TaskCriticality.CRITICAL,
}

# (RuntimeMode, TaskCriticality) → ModelTier
ROUTING_MATRIX: dict[tuple[RuntimeMode, TaskCriticality], ModelTier] = {
    # EXPLORATION
    (RuntimeMode.EXPLORATION, TaskCriticality.LOW): ModelTier.FAST,
    (RuntimeMode.EXPLORATION, TaskCriticality.MEDIUM): ModelTier.BALANCED,
    (RuntimeMode.EXPLORATION, TaskCriticality.HIGH): ModelTier.BALANCED,
    (RuntimeMode.EXPLORATION, TaskCriticality.CRITICAL): ModelTier.FLAGSHIP_HIGH,
    # STANDARD
    (RuntimeMode.STANDARD, TaskCriticality.LOW): ModelTier.FAST,
    (RuntimeMode.STANDARD, TaskCriticality.MEDIUM): ModelTier.BALANCED,
    (RuntimeMode.STANDARD, TaskCriticality.HIGH): ModelTier.BALANCED,
    (RuntimeMode.STANDARD, TaskCriticality.CRITICAL): ModelTier.FLAGSHIP_HIGH,
    # FOCUS
    (RuntimeMode.FOCUS, TaskCriticality.LOW): ModelTier.FAST,
    (RuntimeMode.FOCUS, TaskCriticality.MEDIUM): ModelTier.BALANCED,
    (RuntimeMode.FOCUS, TaskCriticality.HIGH): ModelTier.BALANCED,
    (RuntimeMode.FOCUS, TaskCriticality.CRITICAL): ModelTier.FLAGSHIP_HIGH,
    # MODEL_FREEZE
    (RuntimeMode.MODEL_FREEZE, TaskCriticality.LOW): ModelTier.FAST,
    (RuntimeMode.MODEL_FREEZE, TaskCriticality.MEDIUM): ModelTier.BALANCED,
    (RuntimeMode.MODEL_FREEZE, TaskCriticality.HIGH): ModelTier.BALANCED,
    (RuntimeMode.MODEL_FREEZE, TaskCriticality.CRITICAL): ModelTier.FLAGSHIP_HIGH,
    # SUBMISSION_MODE
    (RuntimeMode.SUBMISSION_MODE, TaskCriticality.LOW): ModelTier.FAST,
    (RuntimeMode.SUBMISSION_MODE, TaskCriticality.MEDIUM): ModelTier.FAST,
    (RuntimeMode.SUBMISSION_MODE, TaskCriticality.HIGH): ModelTier.BALANCED,
    (RuntimeMode.SUBMISSION_MODE, TaskCriticality.CRITICAL): ModelTier.FLAGSHIP_HIGH,
}


# ═══════════════════════════════════════════════════════════════
# Competition Routing Policy
# ═══════════════════════════════════════════════════════════════

class CompetitionRoutingPolicy:
    """Deterministic deadline-aware routing policy.

    Input: task profile + runtime mode + criticality + budget
    Output: RoutingDecision with capability tier, max attempts, etc.
    """

    def __init__(self, matrix: Optional[dict] = None):
        self._matrix = matrix or dict(ROUTING_MATRIX)

    def route(
        self,
        agent: str,
        task_type: str,
        runtime_mode: RuntimeMode,
        criticality: Optional[TaskCriticality] = None,
        previous_attempts: int = 0,
        previous_schema_failures: int = 0,
        budget_available: bool = True,
        is_critical_task: bool = False,
        can_use_critical_reserve: bool = False,
    ) -> RoutingDecision:
        """Produce a routing decision."""
        if criticality is None:
            criticality = TASK_CRITICALITY.get(task_type, TaskCriticality.MEDIUM)

        # Base tier from matrix
        tier = self._matrix.get(
            (runtime_mode, criticality),
            ModelTier.BALANCED,
        )

        rationale = []

        # Escalation: schema failures → upgrade
        if previous_schema_failures >= 2:
            if tier == ModelTier.FAST:
                tier = ModelTier.BALANCED
            elif tier == ModelTier.BALANCED:
                tier = ModelTier.FLAGSHIP_HIGH
            rationale.append("PREVIOUS_SCHEMA_FAILURE")

        # Budget constraint
        if not budget_available:
            if is_critical_task and can_use_critical_reserve:
                rationale.append("CRITICAL_RESERVE")
            elif not is_critical_task:
                rationale.append("LOW_BUDGET_BLOCK")
                tier = ModelTier.FAST  # downgrade

        # Runtime mode rationale
        if runtime_mode == RuntimeMode.MODEL_FREEZE:
            rationale.append("MODEL_FREEZE")
        elif runtime_mode == RuntimeMode.SUBMISSION_MODE:
            rationale.append("SUBMISSION_MODE")

        if criticality == TaskCriticality.CRITICAL:
            rationale.append("CRITICAL_TASK")

        # Max attempts
        max_attempts = 3
        if runtime_mode == RuntimeMode.MODEL_FREEZE:
            max_attempts = 2
        elif runtime_mode == RuntimeMode.SUBMISSION_MODE:
            max_attempts = 1

        # Parallelism
        allow_parallel = runtime_mode in (
            RuntimeMode.EXPLORATION, RuntimeMode.STANDARD,
        )
        parallelism_limit = 3 if allow_parallel else 1

        return RoutingDecision(
            agent=agent,
            task_type=task_type,
            criticality=criticality,
            runtime_mode=runtime_mode,
            capability_tier=tier,
            max_attempts=max_attempts,
            allow_parallel=allow_parallel,
            parallelism_limit=parallelism_limit,
            rationale_codes=rationale,
            escalation_policy="escalate_on_schema_failure" if previous_schema_failures < 2 else "none",
            fallback_policy="strict" if runtime_mode == RuntimeMode.SUBMISSION_MODE else "none",
        )