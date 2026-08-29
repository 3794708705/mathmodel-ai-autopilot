"""MathModel AI — Problem Analysis domain model.

Structured output of ProblemAgent: the formal understanding
of a mathematical modeling competition problem.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from mathmodel.domain.evidence import EvidenceItem


class ModelingTaskType(str, Enum):
    """Classification of modeling tasks a subproblem may require."""
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"
    CLASSIFICATION = "classification"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    SIMULATION = "simulation"
    DECISION_ANALYSIS = "decision_analysis"
    GRAPH_NETWORK = "graph_network"
    DYNAMIC_SYSTEM = "dynamic_system"
    SCHEDULING = "scheduling"
    ROUTING = "routing"
    MULTI_OBJECTIVE = "multi_objective"
    TIME_SERIES = "time_series"
    MIXED = "mixed"


class AmbiguityImpact(str, Enum):
    """Severity of an ambiguity."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Ambiguity(BaseModel):
    """A detected ambiguity in the problem statement.

    Ambiguities are recorded with multiple interpretations.
    Only HIGH/CRITICAL unresolvable ambiguities require human review.
    """

    ambiguity_id: str = Field(default_factory=lambda: f"AMB-{uuid4().hex[:8]}")
    description: str = Field(..., min_length=1)
    interpretations: list[str] = Field(default_factory=list)
    preferred_interpretation: Optional[str] = None
    preference_reason: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    impact: AmbiguityImpact = AmbiguityImpact.MEDIUM
    requires_human_review: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Subproblem(BaseModel):
    """A decomposed subproblem from the main problem.

    Each subproblem may require different modeling approaches.
    Task types can be multiple (e.g., prediction + optimization).
    """

    subproblem_id: str = Field(default_factory=lambda: f"SUB-{uuid4().hex[:8]}")
    original_text: str = Field(..., min_length=1)
    normalized_goal: str = Field(..., min_length=1)
    task_types: list[ModelingTaskType] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(
        default_factory=list,
        description="subproblem_ids this depends on",
    )
    evaluation_target: Optional[str] = Field(
        default=None, description="How success is measured for this subproblem"
    )
    ambiguity_ids: list[str] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, description="1 = highest priority")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProblemAnalysis(BaseModel):
    """Complete problem understanding from ProblemAgent.

    This is the structured domain model — not just human-readable text.
    All subsequent agents consume this structured data.
    """

    analysis_id: str = Field(default_factory=lambda: f"PA-{uuid4().hex[:8]}")

    # Core understanding
    background: str = Field(..., min_length=1)
    core_problem: str = Field(..., min_length=1)
    objectives: list[str] = Field(default_factory=list)

    # Decomposition
    subproblems: list[Subproblem] = Field(default_factory=list)

    # Evidence
    evidence: list[EvidenceItem] = Field(default_factory=list)

    # Constraints
    explicit_constraints: list[str] = Field(default_factory=list)
    implicit_conditions: list[str] = Field(default_factory=list)

    # Ambiguities
    ambiguities: list[Ambiguity] = Field(default_factory=list)

    # Traps and warnings
    possible_traps: list[str] = Field(default_factory=list)

    # Task classification
    modeling_tasks: list[ModelingTaskType] = Field(default_factory=list)

    # Unresolved
    unresolved_questions: list[str] = Field(default_factory=list)

    # Meta
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: Optional[str] = Field(default=None, description="Human-readable summary")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_references(self) -> "ProblemAnalysis":
        """Validate that all references are consistent."""
        evidence_ids = {e.evidence_id for e in self.evidence}
        subproblem_ids = {s.subproblem_id for s in self.subproblems}
        ambiguity_ids = {a.ambiguity_id for a in self.ambiguities}

        errors = []

        # Check subproblem dependency references
        for sp in self.subproblems:
            for dep_id in sp.dependencies:
                if dep_id not in subproblem_ids:
                    errors.append(
                        f"Subproblem {sp.subproblem_id} references unknown "
                        f"dependency {dep_id}"
                    )
            for amb_id in sp.ambiguity_ids:
                if amb_id not in ambiguity_ids:
                    errors.append(
                        f"Subproblem {sp.subproblem_id} references unknown "
                        f"ambiguity {amb_id}"
                    )

        # Check evidence derivation references
        for ev in self.evidence:
            for ref_id in ev.derived_from:
                if ref_id not in evidence_ids:
                    errors.append(
                        f"Evidence {ev.evidence_id} references unknown "
                        f"derivation source {ref_id}"
                    )

        if errors:
            raise ValueError("Reference validation failed: " + "; ".join(errors))

        return self

    def validate_domain(self) -> list[str]:
        """Validate all domain invariants. Returns list of issues."""
        issues = []

        # Unique IDs
        ev_ids = [e.evidence_id for e in self.evidence]
        if len(ev_ids) != len(set(ev_ids)):
            issues.append("Duplicate evidence IDs detected")

        sub_ids = [s.subproblem_id for s in self.subproblems]
        if len(sub_ids) != len(set(sub_ids)):
            issues.append("Duplicate subproblem IDs detected")

        amb_ids = [a.ambiguity_id for a in self.ambiguities]
        if len(amb_ids) != len(set(amb_ids)):
            issues.append("Duplicate ambiguity IDs detected")

        # Evidence invariants
        for ev in self.evidence:
            issues.extend(
                f"Evidence {ev.evidence_id}: {v}"
                for v in ev.validate_invariants()
            )

        # Subproblem task types
        valid_tasks = set(ModelingTaskType)
        for sp in self.subproblems:
            for tt in sp.task_types:
                if tt not in valid_tasks:
                    issues.append(
                        f"Subproblem {sp.subproblem_id}: invalid task type {tt}"
                    )

        return issues