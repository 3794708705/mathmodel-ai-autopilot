"""MathModel AI — Model Jury domain model.

Scores and ranks candidate models using weighted criteria.
The jury computation is deterministic — LLM provides dimension scores,
Python computes the weighted total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class JuryDimension(str, Enum):
    """Scoring dimensions for model evaluation."""
    PROBLEM_FIT = "problem_fit"
    DATA_FIT = "data_fit"
    MATHEMATICAL_VALIDITY = "mathematical_validity"
    EXPLAINABILITY = "explainability"
    VALIDATION_POTENTIAL = "validation_potential"
    INNOVATION = "innovation"
    COMPETITION_FEASIBILITY = "competition_feasibility"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"


# Default weights sum to 100
DEFAULT_JURY_WEIGHTS: dict[JuryDimension, float] = {
    JuryDimension.PROBLEM_FIT: 25.0,
    JuryDimension.DATA_FIT: 15.0,
    JuryDimension.MATHEMATICAL_VALIDITY: 15.0,
    JuryDimension.EXPLAINABILITY: 10.0,
    JuryDimension.VALIDATION_POTENTIAL: 10.0,
    JuryDimension.INNOVATION: 10.0,
    JuryDimension.COMPETITION_FEASIBILITY: 10.0,
    JuryDimension.COMPUTATIONAL_EFFICIENCY: 5.0,
}


class JuryScore(BaseModel):
    """Score for a single candidate across all dimensions.

    LLM provides raw_scores and reasoning.
    Python computes weighted_scores and total_score.
    """

    candidate_id: str
    raw_scores: dict[JuryDimension, float] = Field(
        default_factory=dict,
        description="Raw scores 0-100 per dimension (from LLM)",
    )
    reasoning: dict[JuryDimension, str] = Field(
        default_factory=dict,
        description="LLM reasoning per dimension",
    )

    # Computed by Python (not from LLM)
    weights: dict[JuryDimension, float] = Field(
        default_factory=dict,
        description="Weights used for scoring",
    )
    weighted_scores: dict[JuryDimension, float] = Field(
        default_factory=dict,
        description="raw_score * weight / 100",
    )
    total_score: float = Field(default=0.0, description="Sum of weighted scores")

    @model_validator(mode="after")
    def validate_scores(self) -> "JuryScore":
        """Validate that all raw scores are in range 0-100."""
        for dim, score in self.raw_scores.items():
            if score < 0 or score > 100:
                raise ValueError(
                    f"Score for {dim.value} is {score}, must be 0-100"
                )
        return self


class SelectionOverride(BaseModel):
    """Documented override when the jury selects a non-top-scoring candidate."""

    original_top_candidate: str
    override_candidate: str
    reason: str = Field(..., min_length=1)
    impact: str = Field(..., min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ModelJuryResult(BaseModel):
    """Complete jury evaluation result.

    Ranks all eligible candidates and selects the primary and backup models.
    """

    result_id: str = Field(default_factory=lambda: f"JURY-{uuid4().hex[:8]}")

    # Scores per candidate
    candidate_scores: list[JuryScore] = Field(default_factory=list)

    # Ranking (candidate_ids in order, best first)
    ranking: list[str] = Field(default_factory=list)

    # Selection
    selected_model: Optional[str] = None
    backup_model: Optional[str] = None
    rejected_models: list[str] = Field(default_factory=list)

    # Decision
    decision_reason: str = Field(default="", description="Human-readable decision rationale")
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Override
    selection_override: Optional[SelectionOverride] = None

    # Meta
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selection(self) -> "ModelJuryResult":
        """Validate that selected and backup are in the ranking."""
        if self.selected_model and self.selected_model not in self.ranking:
            raise ValueError(
                f"selected_model {self.selected_model} is not in ranking"
            )
        if self.backup_model and self.backup_model not in self.ranking:
            raise ValueError(
                f"backup_model {self.backup_model} is not in ranking"
            )
        if self.selected_model and self.backup_model:
            if self.selected_model == self.backup_model:
                raise ValueError("selected_model and backup_model must differ")
        return self


def compute_jury_scores(
    candidate_id: str,
    raw_scores: dict[JuryDimension, float],
    reasoning: dict[JuryDimension, str],
    weights: Optional[dict[JuryDimension, float]] = None,
) -> JuryScore:
    """Compute weighted scores deterministically from raw LLM scores.

    LLM is never trusted to compute the total — Python does it.
    """
    w = weights or DEFAULT_JURY_WEIGHTS

    # Validate all dimensions are present
    missing_dims = [dim for dim in JuryDimension if dim not in w]
    if missing_dims:
        raise ValueError(
            f"Weights missing dimensions: {[d.value for d in missing_dims]}. "
            f"All 8 JuryDimension values must be present."
        )

    # Validate weights sum to 100
    total_weight = sum(w.values())
    if abs(total_weight - 100.0) > 0.01:
        raise ValueError(f"Weights sum to {total_weight}, expected 100")

    weighted = {}
    total = 0.0
    for dim in JuryDimension:
        raw = raw_scores.get(dim, 0.0)
        weight = w.get(dim, 0.0)
        ws = raw * weight / 100.0
        weighted[dim] = ws
        total += ws

    return JuryScore(
        candidate_id=candidate_id,
        raw_scores=raw_scores,
        reasoning=reasoning,
        weights=w,
        weighted_scores=weighted,
        total_score=round(total, 2),
    )