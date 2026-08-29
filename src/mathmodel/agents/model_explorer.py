"""MathModel AI — ModelExplorer agent.

Generates diverse candidate mathematical models for a problem.
Enforces candidate diversity: highly similar candidates are rejected.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from pydantic import BaseModel, Field

from mathmodel.agents.base import (
    AgentError,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from mathmodel.domain.analysis import ProblemAnalysis, Subproblem
from mathmodel.domain.candidates import (
    ModelCandidate,
    ModelComponent,
    ModelFamily,
    ComponentRole,
)
from mathmodel.domain.state_helpers import (
    load_analysis,
    record_revision,
    store_candidates,
)
from mathmodel.models.problem_state import ProblemState, ProblemStateStage
from mathmodel.routing.profile import TaskProfile, TaskType
from mathmodel.routing.router import ModelRouter

logger = logging.getLogger(__name__)


class CandidateListOutput(BaseModel):
    """Wrapper for a list of model candidates."""
    candidates: list[ModelCandidate]
    generation_note: str = Field(
        default="",
        description="Required if fewer than 3 candidates: explain why",
    )


class ModelExplorer(BaseAgent):
    """Generates diverse candidate mathematical models.

    Input: ProblemAnalysis + current data availability + competition context
    Output: 3-5 ModelCandidates (fewer only with explicit justification)
    """

    name = "ModelExplorer"
    role = "Candidate model generation and exploration"
    input_schema = BaseModel  # Uses ProblemAnalysis directly
    output_schema = CandidateListOutput
    capabilities = [
        "model_generation",
        "candidate_exploration",
        "diversity_checking",
    ]

    # Minimum required candidates (unless generation_note explains why)
    MIN_CANDIDATES = 3
    # Maximum candidates
    MAX_CANDIDATES = 5

    def __init__(self, router: ModelRouter):
        super().__init__()
        self._router = router

    async def run(self, state: ProblemState) -> AgentResult:
        result = self._start_result()

        try:
            analysis = load_analysis(state)
            if analysis is None:
                return self._finish_result(
                    result,
                    AgentStatus.FAILED,
                    errors=[AgentError(
                        message="No ProblemAnalysis found in state. Run ProblemAgent first.",
                        error_type="prerequisite",
                    )],
                )

            prompt = self._build_prompt(analysis, state)
            profile = TaskProfile.for_task_type(TaskType.MODEL_EXPLORATION)

            logger.info("ModelExplorer requesting candidate generation")
            output = await self._router.route_structured_generate(
                profile=profile,
                prompt=prompt,
                output_schema=CandidateListOutput,
                system_prompt=self._system_prompt(),
            )

            if not isinstance(output, CandidateListOutput):
                return self._finish_result(
                    result,
                    AgentStatus.FAILED,
                    errors=[AgentError(
                        message=f"Expected CandidateListOutput, got {type(output).__name__}",
                        error_type="schema",
                    )],
                )

            # Validate candidates
            errors = self._validate_candidates(output.candidates)
            # Cross-object validation: candidate subproblems must reference real subproblems
            if analysis:
                subproblem_ids = {sp.subproblem_id for sp in analysis.subproblems}
                for c in output.candidates:
                    for sp_id in c.applicable_subproblems:
                        if sp_id not in subproblem_ids:
                            errors.append(AgentError(
                                message=f"Candidate {c.candidate_id} references unknown "
                                        f"subproblem {sp_id}",
                                error_type="cross_reference",
                            ))
            if errors:
                return self._finish_result(
                    result,
                    AgentStatus.FAILED,
                    errors=errors,
                )

            # Check minimum candidate count
            if len(output.candidates) < self.MIN_CANDIDATES:
                if not output.generation_note:
                    return self._finish_result(
                        result,
                        AgentStatus.FAILED,
                        errors=[AgentError(
                            message=f"Only {len(output.candidates)} candidates generated "
                                    f"(minimum {self.MIN_CANDIDATES}) without generation_note",
                            error_type="count",
                        )],
                    )

            # Check diversity
            diversity_issues = self._check_diversity(output.candidates)
            if diversity_issues:
                return self._finish_result(
                    result,
                    AgentStatus.FAILED,
                    errors=[AgentError(
                        message=f"Candidate diversity check failed: {'; '.join(diversity_issues)}",
                        error_type="diversity",
                        details={"issues": diversity_issues},
                    )],
                )

            # Store in ProblemState
            store_candidates(state, output.candidates)
            record_revision(
                state,
                self.name,
                ProblemStateStage.EXPLORE,
                f"Generated {len(output.candidates)} candidates",
            )

            state.current_stage = ProblemStateStage.EXPLORE

            return self._finish_result(
                result,
                AgentStatus.COMPLETED,
                output={
                    "candidate_count": len(output.candidates),
                    "candidate_ids": [c.candidate_id for c in output.candidates],
                },
            )

        except Exception as e:
            logger.exception("ModelExplorer failed")
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
        if not isinstance(output, CandidateListOutput):
            return [AgentError(
                message=f"Expected CandidateListOutput, got {type(output).__name__}",
                error_type="schema",
            )]
        return self._validate_candidates(output.candidates)

    def _validate_candidates(self, candidates: list[ModelCandidate]) -> list[AgentError]:
        errors = []

        # Unique IDs
        ids = [c.candidate_id for c in candidates]
        if len(ids) != len(set(ids)):
            errors.append(AgentError(
                message="Duplicate candidate IDs detected",
                error_type="validation",
                details={"ids": ids},
            ))

        for c in candidates:
            # Component ID uniqueness within candidate
            comp_ids = [comp.component_id for comp in c.components]
            if len(comp_ids) != len(set(comp_ids)):
                errors.append(AgentError(
                    message=f"Candidate {c.candidate_id}: duplicate component IDs",
                    error_type="validation",
                ))

            # Component dependency references
            for comp in c.components:
                for dep_id in comp.dependencies:
                    if dep_id not in comp_ids:
                        errors.append(AgentError(
                            message=f"Candidate {c.candidate_id}: component "
                                    f"{comp.component_id} references unknown "
                                    f"dependency {dep_id}",
                            error_type="reference",
                        ))

            # Cycle detection
            if c.has_cycles:
                cycles = c._find_cycles()
                errors.append(AgentError(
                    message=f"Candidate {c.candidate_id}: component dependency "
                            f"graph has cycles: {cycles}",
                    error_type="cycle",
                    details={"cycles": cycles},
                ))

        return errors

    def _check_diversity(self, candidates: list[ModelCandidate]) -> list[str]:
        """Check that candidates are sufficiently diverse.

        Returns a list of issue descriptions (empty = good).
        Uses is_essentially_same_as() for deep comparison.
        """
        issues = []

        # Group by similarity key
        groups: dict[str, list[ModelCandidate]] = {}
        for c in candidates:
            key = c.similarity_key()
            groups.setdefault(key, []).append(c)

        # Check for overly similar groups
        for key, group in groups.items():
            if len(group) > 2:
                issues.append(
                    f"Too many similar candidates ({len(group)}) with "
                    f"similarity key '{key}'. Candidates: "
                    f"{[c.name for c in group]}"
                )

        # Pairwise deep comparison
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if candidates[i].is_essentially_same_as(candidates[j]):
                    issues.append(
                        f"Candidates '{candidates[i].name}' and "
                        f"'{candidates[j].name}' are essentially the same model. "
                        f"Same family, same structure, or same assumptions."
                    )

        return issues

    def _build_prompt(
        self, analysis: ProblemAnalysis, state: ProblemState
    ) -> str:
        parts = [
            "You are exploring candidate mathematical models for a competition problem.",
            "",
            "## PROBLEM ANALYSIS",
            f"Background: {analysis.background}",
            f"Core problem: {analysis.core_problem}",
            f"Objectives: {', '.join(analysis.objectives)}",
            "",
            "## SUBPROBLEMS",
        ]

        for sp in analysis.subproblems:
            parts.append(
                f"- {sp.subproblem_id}: {sp.normalized_goal} "
                f"[tasks: {', '.join(t.value for t in sp.task_types)}]"
            )

        parts.extend([
            "",
            "## CONSTRAINTS",
            "Explicit: " + "; ".join(analysis.explicit_constraints),
            "Implicit: " + "; ".join(analysis.implicit_conditions),
            "",
            "## INSTRUCTIONS",
            f"Generate {self.MIN_CANDIDATES}-{self.MAX_CANDIDATES} diverse candidate models.",
            "Each candidate must be GENUINELY DIFFERENT from the others — "
            "different model families, different mathematical structures, "
            "different assumptions. Do NOT produce variations of the same model.",
            "",
            "For each candidate, provide:",
            "- name: Descriptive name",
            "- model_family: One of the ModelFamily enum values",
            "- applicable_subproblems: Which subproblem_ids this addresses",
            "- summary: 2-3 sentence description",
            "- mathematical_structure: Brief mathematical formulation",
            "- components: Model components (at least one CORE_MODEL component)",
            "- required_data: What data is needed",
            "- required_assumptions: What assumptions are needed",
            "- strengths: Why this model is good for this problem",
            "- weaknesses: Limitations of this approach",
            "- implementation_plan: How to implement this model",
            "- validation_plan: How to validate results",
            "- risk_flags: Any risks or concerns",
            "",
            "CRITICAL RULES:",
            "- Every candidate must have a unique candidate_id",
            "- Every component must have a unique component_id",
            "- Do NOT fabricate numerical results",
            "- Do NOT claim solver performance you haven't measured",
            "- If fewer than 3 candidates are truly appropriate, explain why in generation_note",
            "- Candidates must be diverse — do not generate MILP, weighted MILP, improved MILP as three candidates",
        ])

        return "\n".join(parts)

    def _system_prompt(self) -> str:
        return (
            "You are an expert in mathematical modeling for competitions. "
            "You know when to apply linear programming, integer programming, "
            "network flow, regression, time series, simulation, and other techniques. "
            "You generate genuinely diverse candidate models, not minor variations. "
            "You never fabricate results or performance claims."
        )