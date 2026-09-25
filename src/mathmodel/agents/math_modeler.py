"""MathModel AI — MathModeler agent.

Converts a selected candidate model into a formal MathematicalModel
with variables, parameters, equations, objectives, and constraints.
"""

from __future__ import annotations

import logging
from typing import Optional, Type

from pydantic import BaseModel, Field

from mathmodel.agents.base import AgentError, AgentResult, AgentStatus, BaseAgent
from mathmodel.domain.math_model import (
    KIND_DUPLICATE_SYMBOL,
    KIND_SYMBOL_COLLISION,
    KIND_UNKNOWN_DEPENDENCY,
    KIND_UNKNOWN_PARAMETER,
    KIND_UNKNOWN_VARIABLE,
    MathematicalModel,
    Parameter,
    Objective,
    Constraint,
    Equation,
    SymbolClosureIssue,
    parse_closure_issues,
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

    async def run(
        self,
        state: ProblemState,
        external_feedback: Optional[list[str]] = None,
    ) -> AgentResult:
        result = self._start_result()

        try:
            analysis = load_analysis(state)
            if not analysis:
                return self._finish_result(result, AgentStatus.FAILED,
                    errors=[AgentError(message="No ProblemAnalysis found", error_type="prerequisite")])

            selected = state.selected_model or {}
            candidate_id = selected.get("candidate_id", "unknown")

            prompt = self._build_prompt(analysis, state)
            profile = TaskProfile.for_task_type(TaskType.MATHEMATICAL_MODELING)

            # Attempt with corrective feedback
            max_attempts = 3
            # Feedback from outside the agent (e.g. an independent verification
            # showing the model contradicts the problem) is as actionable as a
            # schema error, so seed the first attempt with it.
            previous_errors = list(external_feedback) if external_feedback else None
            for attempt in range(1, max_attempts + 1):
                attempt_prompt = prompt
                if previous_errors:
                    attempt_prompt = self._corrective_prompt(prompt, previous_errors)

                try:
                    output = await self._router.route_structured_generate(
                        profile=profile,
                        prompt=attempt_prompt,
                        output_schema=MathematicalModel,
                        system_prompt=self._system_prompt(),
                    )
                except Exception as e:
                    # A schema or parse failure is also worth a corrective retry:
                    # the previous response tells us what the model got wrong.
                    if attempt < max_attempts:
                        # A symbol-closure failure names the exact symbols that
                        # were never declared. Handing those back lets the next
                        # attempt declare or drop precisely those identifiers
                        # instead of regenerating the whole model and repeating
                        # the same mistake.
                        issues = parse_closure_issues(str(e))
                        if issues:
                            previous_errors = self._symbol_closure_feedback(issues)
                            logger.warning(
                                "MathModeler attempt %d failed symbol closure "
                                "(%d undeclared symbol reference(s)); "
                                "sending the exact symbol list back",
                                attempt, len(issues),
                            )
                        else:
                            previous_errors = [
                                f"Response could not be parsed as a valid "
                                f"MathematicalModel: {str(e)[:600]}"
                            ]
                            logger.warning(
                                "MathModeler attempt %d failed schema validation: %s",
                                attempt, e,
                            )
                        continue
                    raise

                if not isinstance(output, MathematicalModel):
                    if attempt < max_attempts:
                        previous_errors = [f"Schema validation: expected MathematicalModel, got {type(output).__name__}"]
                        continue
                    return self._finish_result(result, AgentStatus.FAILED,
                        errors=[AgentError(message=f"Expected MathematicalModel, got {type(output).__name__}", error_type="schema")])

                issues = output.validate_domain()
                if not issues:
                    break  # success

                if attempt < max_attempts:
                    previous_errors = issues
                else:
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

    @staticmethod
    def _symbol_closure_feedback(issues: list[SymbolClosureIssue]) -> list[str]:
        """Turn undeclared-symbol issues into an exact, targeted repair list."""
        def names(kind: str) -> list[str]:
            return sorted({i.symbol for i in issues if i.kind == kind})

        unknown_variables = names(KIND_UNKNOWN_VARIABLE)
        unknown_parameters = names(KIND_UNKNOWN_PARAMETER)
        unknown_dependencies = names(KIND_UNKNOWN_DEPENDENCY)
        duplicates = names(KIND_DUPLICATE_SYMBOL)
        collisions = names(KIND_SYMBOL_COLLISION)

        lines = [
            "SYMBOL CLOSURE FAILURE — the model was REJECTED because equations "
            "reference identifiers that are never declared. This is the only "
            "problem to fix.",
        ]
        if unknown_variables:
            lines.append(
                "Unknown VARIABLES (listed in an equation's `variable_ids` but "
                "absent from `variables`): " + ", ".join(unknown_variables)
            )
        if unknown_parameters:
            lines.append(
                "Unknown PARAMETERS (listed in an equation's `parameter_ids` but "
                "absent from `parameters`): " + ", ".join(unknown_parameters)
            )
        if unknown_dependencies:
            lines.append(
                "Unknown DEPENDENCIES (listed in an equation's `dependencies` but "
                "matching no `equation_id`): " + ", ".join(unknown_dependencies)
            )
        if duplicates:
            lines.append(
                "DUPLICATE variable symbols (declared more than once): "
                + ", ".join(duplicates)
            )
        if collisions:
            lines.append(
                "SYMBOL COLLISIONS (used as both a variable symbol and a "
                "parameter symbol): " + ", ".join(collisions)
            )
        lines.extend([
            "",
            "For each symbol above, apply EXACTLY ONE of these two fixes:",
            "  (a) declare it — add an entry to `variables` (or `parameters`) "
            "whose `variable_id`/`parameter_id` is EXACTLY that string, and whose "
            "`symbol` is also that string; or",
            "  (b) remove it — delete that string from the offending equation's "
            "`variable_ids` / `parameter_ids` / `dependencies` list, if the "
            "equation does not actually need it.",
            "",
            "Keep every other part of the model as it is. Do not rename symbols "
            "that are already declared, do not rebuild the model from scratch, "
            "and do not drop equations that are otherwise correct. Re-emit the "
            "whole model with only these symbols corrected.",
        ])
        return lines

    def _build_prompt(self, analysis, state: ProblemState) -> str:
        # The modeler must see the real problem statement, the real extracted
        # facts and the real attachment schema. Without them it cannot know
        # which parameters are already given and marks everything REQUIRED.
        facts = "\n".join(
            f"- [{e.type.value}] {e.content}" for e in analysis.evidence
        ) or "(none extracted)"

        parts = [
            "Build a formal mathematical model for a competition problem.",
            "## PROBLEM STATEMENT (verbatim, includes attachment schema)",
            (state.raw_problem or "(no problem text provided)")[:20000],
            "",
            f"## CORE PROBLEM: {analysis.core_problem}",
            f"## OBJECTIVES: {', '.join(analysis.objectives)}",
            "## EXTRACTED FACTS AND DATA",
            facts,
            "## CONSTRAINTS",
            f"Explicit: {'; '.join(analysis.explicit_constraints)}",
            f"Implicit: {'; '.join(analysis.implicit_conditions)}",
            "",
            "## INSTRUCTIONS",
            "Produce a complete MathematicalModel with:",
            "- variables: decision variables, state variables (with symbols, types, bounds, units)",
            "- parameters: known parameters from data/problem (with values if known, REQUIRED if missing)",
            "- objectives: minimize/maximize expressions with meaning",
            "  Each objective MUST also carry `priority`, an INTEGER >= 1 "
            "(1 = most important). Never use 0 or a negative value.",
            "- constraints: all constraints with relation (eq/le/ge), rhs, meaning",
            "- equations: formal equations with derivation, variable references",
            "- algorithm_plan: how to solve this model",
            "- solver_requirements: what solver capabilities are needed",
            "",
            "CRITICAL RULES:",
            "- Do NOT fabricate parameter values — use status 'required' only when the "
            "problem and data truly do not provide the value",
            "- `status` MUST be exactly one of the lowercase enum values: "
            "'known', 'estimated', 'required', 'calibrated'. Never write 'REQUIRED'.",
            "- Every value that appears in the problem statement or in the extracted "
            "facts/data above is 'known' and MUST carry that numeric value "
            "(e.g. a stated count, a stated granularity, a stated limit)",
            "- `constraint.rhs` MUST be a plain number. Put any symbolic right-hand "
            "side inside `constraint.expression` instead, e.g. 'x_i + y_i <= T_h'.",
            "- Every symbol must be unique across variables and parameters",
            "- SYMBOL CLOSURE: every identifier used in an objective, constraint or "
            "equation expression MUST be declared in `variables` or `parameters` "
            "first. Before you answer, scan each expression and confirm each symbol "
            "appears in exactly one declaration list. Do not use subscripted or "
            "indexed names such as `delta_f_j` unless that exact string is declared "
            "as a variable or parameter.",
            "- Every equation must reference valid variable_ids and parameter_ids",
            "- Every constraint must have a source (problem statement, data, or assumption)",
            "- Use accepted assumptions only",
            "- Prefer numeric coefficients in expressions (e.g. '3*x1' not 'p_c*x1') "
            "when the parameter value is known",
            "- Keep the model compact: aim for at most 25 variables, 25 parameters, "
            "25 constraints and 12 equations. Use indexed/symbolic forms rather than "
            "one entry per data row.",
        ]
        return "\n".join(parts)

    def _system_prompt(self) -> str:
        return "You are an expert mathematical modeler. You formalize problems into precise mathematical models with well-defined variables, parameters, objectives, and constraints. You never fabricate data."

    @staticmethod
    def _corrective_prompt(base_prompt: str, previous_errors: list[str]) -> str:
        """Add corrective feedback to the prompt for retry."""
        error_text = "\n".join(f"  - {e}" for e in previous_errors)
        return (
            f"{base_prompt}\n\n"
            f"## CORRECTIVE FEEDBACK\n"
            f"The previous attempt had the following errors:\n"
            f"{error_text}\n\n"
            f"Please fix these specific errors in your new model. "
            f"Ensure all parameter IDs referenced in equations are "
            f"defined in the parameters list. Ensure all variable IDs "
            f"referenced in constraints are defined in the variables list. "
            f"Use numeric coefficients directly in expressions when the "
            f"problem provides the values explicitly.\n\n"
            f"## HARD PARAMETER RULES (violating these rejects the model)\n"
            f"- Every parameter must satisfy ONE of these two shapes:\n"
            f"  1. status 'known'  + a numeric `value`.\n"
            f"  2. status 'estimated' + a numeric `value` + a numeric `uncertainty`.\n"
            f"- Never emit status 'required': a model containing an unresolved "
            f"parameter is rejected as incomplete.\n"
            f"- Values that the attachment data determines (counts per class, "
            f"row counts, maxima/minima, interval lengths, the number of "
            f"equipment items) are 'known' — compute them from the data given "
            f"above and put the number in `value`.\n"
            f"- Values the problem statement states are 'known' with that number.\n"
            f"- Only if a value is genuinely absent may you mark it 'estimated', "
            f"and then you MUST also give both `value` and `uncertainty`.\n"
            f"- Every 'estimated' parameter without `uncertainty` is an error; "
            f"either add the uncertainty or change the status to 'known' with "
            f"the concrete value."
        )