"""MathModel AI — Eligibility Gate domain model.

Determines whether a candidate model is eligible for jury evaluation.
Checks hard constraints before the jury spends resources scoring.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EligibilityCheck(str, Enum):
    """Specific eligibility checks performed."""
    REQUIRED_DATA = "required_data"
    MATHEMATICAL_PLAUSIBILITY = "mathematical_plausibility"
    IMPLEMENTATION_PATH = "implementation_path"
    VALIDATION_POSSIBILITY = "validation_possibility"
    COMPETITION_FEASIBILITY = "competition_feasibility"
    HARD_CONSTRAINT_VIOLATION = "hard_constraint_violation"


class EligibilityFailure(BaseModel):
    """A hard failure that makes a candidate ineligible."""
    check: EligibilityCheck
    reason: str = Field(..., min_length=1)
    detail: Optional[str] = None


class EligibilityWarning(BaseModel):
    """A soft warning — candidate is still eligible but has concerns."""
    check: EligibilityCheck
    reason: str = Field(..., min_length=1)
    detail: Optional[str] = None


class EligibilityResult(BaseModel):
    """Result of eligibility checking for a candidate model."""

    result_id: str = Field(default_factory=lambda: f"ELIG-{uuid4().hex[:8]}")
    candidate_id: str
    eligible: bool

    # Failures (hard — make candidate ineligible)
    hard_failures: list[EligibilityFailure] = Field(default_factory=list)

    # Warnings (soft — candidate still eligible)
    warnings: list[EligibilityWarning] = Field(default_factory=list)

    # Detailed assessments
    missing_data: list[str] = Field(default_factory=list)
    assumption_risks: list[str] = Field(default_factory=list)
    implementation_risks: list[str] = Field(default_factory=list)
    validation_risks: list[str] = Field(default_factory=list)
    competition_feasibility: bool = True

    # Summary
    reason: str = Field(default="", description="Human-readable eligibility summary")

    # Meta
    checked_at: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EligibilityPolicy:
    """Policy for determining candidate eligibility.

    Separates hard failures from soft warnings.
    Configurable per competition context.
    """

    def __init__(
        self,
        strict_data_requirements: bool = True,
        strict_math_plausibility: bool = True,
        allow_high_computation: bool = True,
        allow_difficult_validation: bool = True,
    ):
        self.strict_data_requirements = strict_data_requirements
        self.strict_math_plausibility = strict_math_plausibility
        self.allow_high_computation = allow_high_computation
        self.allow_difficult_validation = allow_difficult_validation

    def evaluate(
        self,
        candidate_id: str,
        has_required_data: bool,
        math_plausible: bool,
        has_implementation_path: bool,
        can_validate: bool,
        competition_feasible: bool,
        hard_constraint_violations: list[str],
        missing_data: list[str],
        assumption_risks: list[str],
        implementation_risks: list[str],
        validation_risks: list[str],
    ) -> EligibilityResult:
        """Evaluate a candidate and return an eligibility result."""
        hard_failures: list[EligibilityFailure] = []
        warnings: list[EligibilityWarning] = []

        # Required data
        if self.strict_data_requirements and not has_required_data:
            hard_failures.append(EligibilityFailure(
                check=EligibilityCheck.REQUIRED_DATA,
                reason="Required data is not available",
                detail=f"Missing: {', '.join(missing_data)}" if missing_data else None,
            ))

        # Mathematical plausibility
        if self.strict_math_plausibility and not math_plausible:
            hard_failures.append(EligibilityFailure(
                check=EligibilityCheck.MATHEMATICAL_PLAUSIBILITY,
                reason="Mathematical premise is not plausible",
            ))

        # Hard constraint violations
        for violation in hard_constraint_violations:
            hard_failures.append(EligibilityFailure(
                check=EligibilityCheck.HARD_CONSTRAINT_VIOLATION,
                reason=violation,
            ))

        # Implementation path
        if not has_implementation_path:
            warnings.append(EligibilityWarning(
                check=EligibilityCheck.IMPLEMENTATION_PATH,
                reason="No clear implementation path",
            ))

        # Validation possibility
        if not can_validate:
            if self.allow_difficult_validation:
                warnings.append(EligibilityWarning(
                    check=EligibilityCheck.VALIDATION_POSSIBILITY,
                    reason="Validation may be difficult",
                ))
            else:
                hard_failures.append(EligibilityFailure(
                    check=EligibilityCheck.VALIDATION_POSSIBILITY,
                    reason="Cannot validate this model",
                ))

        # Competition feasibility
        if not competition_feasible:
            hard_failures.append(EligibilityFailure(
                check=EligibilityCheck.COMPETITION_FEASIBILITY,
                reason="Not feasible within competition constraints",
            ))

        # High computation cost — policy controlled
        if implementation_risks and not self.allow_high_computation:
            high_cost_risks = [
                r for r in implementation_risks
                if "comput" in r.lower() or "cost" in r.lower() or "expensive" in r.lower()
            ]
            if high_cost_risks:
                warnings.append(EligibilityWarning(
                    check=EligibilityCheck.IMPLEMENTATION_PATH,
                    reason=f"High computational cost: {'; '.join(high_cost_risks)}",
                ))

        eligible = len(hard_failures) == 0

        return EligibilityResult(
            candidate_id=candidate_id,
            eligible=eligible,
            hard_failures=hard_failures,
            warnings=warnings,
            missing_data=missing_data,
            assumption_risks=assumption_risks,
            implementation_risks=implementation_risks,
            validation_risks=validation_risks,
            competition_feasibility=competition_feasible,
            reason=self._build_reason(eligible, hard_failures, warnings),
        )

    @staticmethod
    def _build_reason(
        eligible: bool,
        hard_failures: list[EligibilityFailure],
        warnings: list[EligibilityWarning],
    ) -> str:
        parts = []
        if eligible:
            parts.append("ELIGIBLE")
        else:
            parts.append("INELIGIBLE")
        if hard_failures:
            parts.append(f"Hard failures: {len(hard_failures)}")
        if warnings:
            parts.append(f"Warnings: {len(warnings)}")
        return ". ".join(parts)