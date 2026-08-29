"""MathModel AI — MathModeler agent.

Converts a selected candidate model into a formal MathematicalModel
with variables, parameters, equations, objectives, and constraints.
"""

from __future__ import annotations

import logging
from typing import Type

from pydantic import BaseModel, Field

from mathmodel.agents.base import AgentError, AgentResult, AgentStatus, BaseAgent
from mathmodel.domain.math_model import (
    MathematicalModel,
    Variable,
    Parameter,
    Objective,
    Constraint,
    Equation,
)
from mathmodel.domain.state_helpers import load_analysis, record_revision
from mathmodel.models.problem_state import ProblemState, ProblemStateStage
from mathmodel.routing.profile import TaskProfile, TaskType
from mathmodel.routing.router import ModelRouter

logger = logging.getLogger(__name__)


class MathModeler(BaseAgent):
    """Builds a formal MathematicalModel from a selected candidate.

    Input: ProblemAnalysis + DataAnalysis + Selected ModelCandidate
    Output: MathematicalModel with variables, parameters, equations
    """

    name = "MathModeler"
    role = "Mathematical model formalization"
    input_schema = BaseModel
    output_schema = MathematicalModel
    capabilities = [
        "variable_definition",
        "parameter_definition",
        "equation_formulation",
        "objective_formulation",
        "constraint_formulation",
        "model_validation",
    ]

    def __init__(self, router: ModelRouter):
        super().__init__()
        self._router = router

    async def run(self, state: ProblemState) -> AgentResult:
        result = self._start_result()

        try:
            analysis = load_analysis(state)
            if not analysis:
                return self._finish_result(result, AgentStatus.FAILED,
                    errors=[AgentError(message="No ProblemAnalysis found", error_type="prerequisite")])

            # Get selected model info
            selected = state.selected_model or {}
            candidate_id = selected.get("candidate_id", "unknown")

            prompt = self._build_prompt(analysis, state)
            profile = TaskProfile.for_task_type(TaskType.MATHEMATICAL_MODELING)

            output = await self._router.route_structured_generate(
                profile=profile,
                prompt=prompt,
                output_schema=MathematicalModel,
                system_prompt=self._system_prompt(),
            )

            if not isinstance(output, MathematicalModel):
                return self._finish_result(result, AgentStatus.FAILED,
                    errors=[AgentError(message=f"Expected MathematicalModel, got {type(output).__name__}", error_type="schema")])

            # Domain validation
            issues = output.validate_domain()
            if issues:
                return self._finish_result(result, AgentStatus.FAILED,
                    errors=[AgentError(message=f"Domain validation: {'; '.join(issues)}", error_type="validation")])

            # Store in state
            if state.metadata_ is None:
                state.metadata_ = {}
            state.metadata_["mathematical_model"] = output.model_dump()
            record_revision(state, self.name, ProblemStateStage.MODEL, f"Model: {output.name}")

            state.current_stage = ProblemStateStage.MODEL

            return self._finish_result(result, AgentStatus.COMPLETED,
                output={"model_id": output.model_id, "variable_count": len(output.variables),
                        "equation_count": len(output.equations)})

        except Exception as e:
            logger.exception("MathModeler failed")
            return self._finish_result(result, AgentStatus.FAILED,
                errors=[AgentError(message=str(e), error_type="runtime", recoverable=True)])

    def validate_output(self, output: BaseModel) -> list[AgentError]:
        if not isinstance(output, MathematicalModel):
            return [AgentError(message=f"Expected MathematicalModel, got {type(output).__name__}", error_type="schema")]
        issues = output.validate_domain()
        return [AgentError(message=i, error_type="validation") for i in issues]

    def _build_prompt(self, analysis, state: ProblemState) -> str:
        parts = [
            "Build a formal mathematical model for a competition problem.",
            f"## PROBLEM: {analysis.core_problem}",
            f"## OBJECTIVES: {', '.join(analysis.objectives)}",
            "## CONSTRAINTS",
            f"Explicit: {'; '.join(analysis.explicit_constraints)}",
            f"Implicit: {'; '.join(analysis.implicit_conditions)}",
            "",
            "## INSTRUCTIONS",
            "Produce a complete MathematicalModel with:",
            "- variables: decision variables, state variables (with symbols, types, bounds, units)",
            "- parameters: known parameters from data/problem (with values if known, REQUIRED if missing)",
            "- objectives: minimize/maximize expressions with meaning",
            "- constraints: all constraints with relation (eq/le/ge), rhs, meaning",
            "- equations: formal equations with derivation, variable references",
            "- algorithm_plan: how to solve this model",
            "- solver_requirements: what solver capabilities are needed",
            "",
            "CRITICAL RULES:",
            "- Do NOT fabricate parameter values — mark as REQUIRED if unknown",
            "- Every symbol must be unique across variables and parameters",
            "- Every equation must reference valid variable_ids and parameter_ids",
            "- Every constraint must have a source (problem statement, data, or assumption)",
            "- Use accepted assumptions only",
        ]
        return "\n".join(parts)

    def _system_prompt(self) -> str:
        return "You are an expert mathematical modeler. You formalize problems into precise mathematical models with well-defined variables, parameters, objectives, and constraints. You never fabricate data."