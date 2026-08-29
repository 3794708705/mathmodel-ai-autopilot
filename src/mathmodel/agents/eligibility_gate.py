"""MathModel AI — Eligibility Gate agent.

Checks whether candidate models are eligible for jury evaluation.
Separates hard failures (ineligible) from soft warnings.
"""

from __future__ import annotations

import logging
from typing import Type

from pydantic import BaseModel

from mathmodel.agents.base import (
    AgentError,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from mathmodel.domain.candidates import ModelCandidate
from mathmodel.domain.eligibility import (
    EligibilityPolicy,
    EligibilityResult,
)
from mathmodel.domain.state_helpers import (
    load_candidates,
    store_eligibility_results,
)
from mathmodel.models.problem_state import ProblemState

logger = logging.getLogger(__name__)


class EligibilityGate(BaseAgent):
    """Checks candidate model eligibility before jury evaluation.

    Applies hard/soft checks: missing data → hard fail,
    expensive computation → warning, etc.
    """

    name = "EligibilityGate"
    role = "Candidate eligibility checking"
    input_schema = BaseModel
    output_schema = BaseModel  # Returns list of EligibilityResult
    capabilities = [
        "eligibility_checking",
        "hard_failure_detection",
        "risk_assessment",
    ]

    def __init__(self, policy: EligibilityPolicy):
        super().__init__()
        self._policy = policy

    async def run(self, state: ProblemState) -> AgentResult:
        result = self._start_result()

        try:
            candidates = load_candidates(state)
            if not candidates:
                return self._finish_result(
                    result,
                    AgentStatus.FAILED,
                    errors=[AgentError(
                        message="No candidates found in state. Run ModelExplorer first.",
                        error_type="prerequisite",
                    )],
                )

            eligibility_results = []
            for candidate in candidates:
                er = self._evaluate_candidate(candidate)
                eligibility_results.append(er)

            # Store results
            store_eligibility_results(state, eligibility_results)

            eligible_count = sum(1 for r in eligibility_results if r.eligible)
            ineligible_count = len(eligibility_results) - eligible_count

            return self._finish_result(
                result,
                AgentStatus.COMPLETED,
                output={
                    "total": len(eligibility_results),
                    "eligible": eligible_count,
                    "ineligible": ineligible_count,
                    "results": [r.model_dump() for r in eligibility_results],
                },
            )

        except Exception as e:
            logger.exception("EligibilityGate failed")
            return self._finish_result(
                result,
                AgentStatus.FAILED,
                errors=[AgentError(
                    message=str(e),
                    error_type="runtime",
                    recoverable=True,
                )],
            )

    def validate_output(self, output: BaseModel) -> list[AgentError]:
        return []

    def _evaluate_candidate(self, candidate: ModelCandidate) -> EligibilityResult:
        """Evaluate a single candidate's eligibility."""
        # Determine checks
        has_required_data = len(candidate.required_data) == 0 or all(
            d for d in candidate.required_data
        )
        math_plausible = candidate.model_family is not None
        has_implementation_path = bool(candidate.implementation_plan)
        can_validate = bool(candidate.validation_plan)
        competition_feasible = True  # Default, could be overridden by risk flags

        if "competition_infeasible" in candidate.risk_flags:
            competition_feasible = False

        hard_constraint_violations = [
            flag for flag in candidate.risk_flags
            if flag.startswith("hard_constraint:")
        ]

        missing_data = [
            d for d in candidate.required_data
            if not d  # Empty string means missing
        ] if not has_required_data else []

        return self._policy.evaluate(
            candidate_id=candidate.candidate_id,
            has_required_data=has_required_data,
            math_plausible=math_plausible,
            has_implementation_path=has_implementation_path,
            can_validate=can_validate,
            competition_feasible=competition_feasible,
            hard_constraint_violations=hard_constraint_violations,
            missing_data=missing_data,
            assumption_risks=list(candidate.required_assumptions),
            implementation_risks=[
                f"Implementation risk: {c.name}"
                for c in candidate.components
                if not c.description
            ],
            validation_risks=[
                "No validation plan" if not candidate.validation_plan else ""
            ],
        )