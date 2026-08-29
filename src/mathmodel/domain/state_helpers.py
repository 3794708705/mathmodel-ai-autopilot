"""MathModel AI — ProblemState domain helpers.

Bridge between typed domain schemas and ProblemState ORM persistence.
All domain objects are validated through Pydantic before storage.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from mathmodel.domain.analysis import ProblemAnalysis
from mathmodel.domain.candidates import ModelCandidate
from mathmodel.domain.eligibility import EligibilityResult
from mathmodel.domain.jury import ModelJuryResult
from mathmodel.domain.literature import LiteratureQueryPlan
from mathmodel.models.problem_state import (
    ProblemState,
    ProblemStateStage,
    ProblemStateStatus,
)


def store_analysis(state: ProblemState, analysis: ProblemAnalysis) -> None:
    """Validate and store a ProblemAnalysis into ProblemState."""
    # Validate domain invariants
    issues = analysis.validate_domain()
    if issues:
        raise ValueError(f"ProblemAnalysis validation failed: {issues}")

    state.background = analysis.background
    state.objectives = analysis.objectives
    state.subproblems = [sp.model_dump() for sp in analysis.subproblems]
    state.facts = [
        ev.model_dump() for ev in analysis.evidence
        if ev.type.value in ("FACT", "DATA")
    ]
    state.assumptions = [
        ev.model_dump() for ev in analysis.evidence
        if ev.is_assumption
    ]
    state.ambiguities = [a.model_dump() for a in analysis.ambiguities]
    state.constraints = [
        {"type": "explicit", "value": c} for c in analysis.explicit_constraints
    ] + [
        {"type": "implicit", "value": c} for c in analysis.implicit_conditions
    ]

    # Store full analysis in metadata for reload
    if state.metadata_ is None:
        state.metadata_ = {}
    state.metadata_["analysis"] = analysis.model_dump()


def load_analysis(state: ProblemState) -> Optional[ProblemAnalysis]:
    """Load a ProblemAnalysis from ProblemState metadata."""
    if state.metadata_ and "analysis" in state.metadata_:
        return ProblemAnalysis.model_validate(state.metadata_["analysis"])
    return None


def store_candidates(state: ProblemState, candidates: list[ModelCandidate]) -> None:
    """Validate and store model candidates into ProblemState."""
    if not candidates:
        raise ValueError("At least one candidate is required")

    # Validate unique IDs
    ids = [c.candidate_id for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs detected")

    state.candidate_models = [c.model_dump() for c in candidates]


def load_candidates(state: ProblemState) -> list[ModelCandidate]:
    """Load model candidates from ProblemState."""
    if not state.candidate_models:
        return []
    return [ModelCandidate.model_validate(c) for c in state.candidate_models]


def store_eligibility_results(
    state: ProblemState, results: list[EligibilityResult]
) -> None:
    """Store eligibility results in ProblemState metadata."""
    if state.metadata_ is None:
        state.metadata_ = {}
    state.metadata_["eligibility_results"] = [r.model_dump() for r in results]


def load_eligibility_results(state: ProblemState) -> list[EligibilityResult]:
    """Load eligibility results from ProblemState."""
    if state.metadata_ and "eligibility_results" in state.metadata_:
        return [
            EligibilityResult.model_validate(r)
            for r in state.metadata_["eligibility_results"]
        ]
    return []


def store_jury_result(state: ProblemState, result: ModelJuryResult) -> None:
    """Store jury result and update selected/backup models."""
    if state.metadata_ is None:
        state.metadata_ = {}
    state.metadata_["jury_result"] = result.model_dump()

    if result.selected_model:
        state.selected_model = {
            "candidate_id": result.selected_model,
            "selected_by": "ModelJury",
            "backup": result.backup_model,
        }
        if result.backup_model:
            state.backup_model = {
                "candidate_id": result.backup_model,
                "selected_by": "ModelJury",
            }


def load_jury_result(state: ProblemState) -> Optional[ModelJuryResult]:
    """Load jury result from ProblemState."""
    if state.metadata_ and "jury_result" in state.metadata_:
        return ModelJuryResult.model_validate(state.metadata_["jury_result"])
    return None


def store_literature_plan(
    state: ProblemState, plan: LiteratureQueryPlan
) -> None:
    """Store literature query plan in ProblemState metadata."""
    if state.metadata_ is None:
        state.metadata_ = {}
    state.metadata_["literature_plan"] = plan.model_dump()


def load_literature_plan(state: ProblemState) -> Optional[LiteratureQueryPlan]:
    """Load literature query plan from ProblemState."""
    if state.metadata_ and "literature_plan" in state.metadata_:
        return LiteratureQueryPlan.model_validate(state.metadata_["literature_plan"])
    return None


def record_revision(
    state: ProblemState,
    agent_name: str,
    stage: ProblemStateStage,
    reason: str,
) -> None:
    """Record a revision in the stage history."""
    history = list(state.stage_history or [])
    revision = len(history) + 1
    history.append({
        "revision": revision,
        "previous_revision": revision - 1 if revision > 1 else None,
        "source_agent": agent_name,
        "stage": stage.value,
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "reason": reason,
    })
    state.stage_history = history