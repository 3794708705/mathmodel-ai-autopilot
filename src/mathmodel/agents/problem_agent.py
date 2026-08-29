"""MathModel AI — ProblemAgent.

Understands a mathematical modeling competition problem.
Produces a structured ProblemAnalysis with evidence, subproblems,
ambiguities, and task classification.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from pydantic import BaseModel

from mathmodel.agents.base import (
    AgentError,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from mathmodel.domain.analysis import (
    Ambiguity,
    AmbiguityImpact,
    ModelingTaskType,
    ProblemAnalysis,
    Subproblem,
)
from mathmodel.domain.evidence import EvidenceItem, EvidenceType
from mathmodel.domain.state_helpers import record_revision, store_analysis
from mathmodel.models.problem_state import ProblemState, ProblemStateStage
from mathmodel.providers.base import GenerationRequest, StructuredGenerationRequest
from mathmodel.routing.profile import TaskProfile, TaskType
from mathmodel.routing.router import ModelRouter

logger = logging.getLogger(__name__)


class ProblemAgentInput(BaseModel):
    """Input schema for ProblemAgent."""
    raw_problem: str = ""
    competition: str = ""
    deadline_info: str = ""


class ProblemAgentOutput(BaseModel):
    """Output schema wrapping ProblemAnalysis."""
    analysis: ProblemAnalysis


class ProblemAgent(BaseAgent):
    """Agent that understands and decomposes a mathematical modeling problem.

    Responsibilities (UNDERSTAND stage only):
    - Problem understanding
    - Subproblem decomposition
    - Evidence extraction (facts, data, assumptions)
    - Explicit constraint extraction
    - Implicit condition identification
    - Ambiguity detection
    - Modeling task classification

    Does NOT handle: model selection, equations, solving, code, paper.
    """

    name = "ProblemAgent"
    role = "Problem understanding and decomposition"
    input_schema = ProblemAgentInput
    output_schema = ProblemAgentOutput
    capabilities = [
        "problem_understanding",
        "subproblem_decomposition",
        "evidence_extraction",
        "constraint_identification",
        "ambiguity_detection",
        "task_classification",
    ]

    def __init__(self, router: ModelRouter):
        super().__init__()
        self._router = router

    async def run(self, state: ProblemState) -> AgentResult:
        result = self._start_result()

        try:
            # Build the prompt
            prompt = self._build_prompt(state)

            # Route to LLM for structured generation
            profile = TaskProfile.for_task_type(TaskType.PROBLEM_UNDERSTANDING)

            logger.info("ProblemAgent requesting structured analysis")
            output = await self._router.route_structured_generate(
                profile=profile,
                prompt=prompt,
                output_schema=ProblemAnalysis,
                system_prompt=self._system_prompt(),
            )

            # Validate the output
            if not isinstance(output, ProblemAnalysis):
                return self._finish_result(
                    result,
                    AgentStatus.FAILED,
                    errors=[AgentError(
                        message=f"Expected ProblemAnalysis, got {type(output).__name__}",
                        error_type="schema",
                    )],
                )

            # Domain validation
            issues = output.validate_domain()
            if issues:
                return self._finish_result(
                    result,
                    AgentStatus.FAILED,
                    errors=[AgentError(
                        message=f"Domain validation failed: {'; '.join(issues)}",
                        error_type="validation",
                        details={"issues": issues},
                    )],
                )

            # Store in ProblemState
            store_analysis(state, output)
            record_revision(
                state,
                self.name,
                ProblemStateStage.UNDERSTAND,
                "Problem analysis completed",
            )

            state.current_stage = ProblemStateStage.UNDERSTAND

            return self._finish_result(
                result,
                AgentStatus.COMPLETED,
                output={"analysis_id": output.analysis_id},
            )

        except Exception as e:
            logger.exception("ProblemAgent failed")
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
        errors = []
        if not isinstance(output, ProblemAnalysis):
            errors.append(AgentError(
                message=f"Output must be ProblemAnalysis, got {type(output).__name__}",
                error_type="schema",
            ))
            return errors

        # Validate domain
        issues = output.validate_domain()
        for issue in issues:
            errors.append(AgentError(
                message=issue,
                error_type="validation",
            ))

        return errors

    def _build_prompt(self, state: ProblemState) -> str:
        """Build the LLM prompt for problem analysis."""
        parts = [
            "You are analyzing a mathematical modeling competition problem.",
            "",
            "## PROBLEM STATEMENT",
            state.raw_problem or "(no problem text provided)",
            "",
        ]

        if state.competition:
            parts.extend([
                "## COMPETITION",
                state.competition,
                "",
            ])

        parts.extend([
            "## INSTRUCTIONS",
            "Analyze the problem and produce a structured ProblemAnalysis with:",
            "",
            "1. **background**: What is the context and domain of this problem?",
            "2. **core_problem**: What is the central question being asked?",
            "3. **objectives**: What specific goals must be achieved?",
            "4. **subproblems**: Decompose into subproblems. For each:",
            "   - original_text: the relevant part of the problem",
            "   - normalized_goal: what this subproblem is trying to achieve",
            "   - task_types: one or more of optimization, prediction, evaluation, classification, statistical_analysis, simulation, decision_analysis, graph_network, dynamic_system, scheduling, routing, multi_objective, mixed",
            "   - inputs, expected_outputs, constraints",
            "   - dependencies: which other subproblems must be solved first",
            "5. **evidence**: Extract facts, data, and proposed assumptions. For each:",
            "   - type: FACT, DATA, or PROPOSED_ASSUMPTION",
            "   - For PROPOSED_ASSUMPTION: include clear justification",
            "   - Never mark an assumption as FACT",
            "6. **explicit_constraints**: Constraints directly stated in the problem",
            "7. **implicit_conditions**: Conditions implied but not explicitly stated",
            "8. **ambiguities**: Where the problem could be interpreted multiple ways. For each:",
            "   - Provide multiple interpretations",
            "   - State your preferred interpretation and why",
            "   - Rate impact: LOW, MEDIUM, HIGH, CRITICAL",
            "9. **possible_traps**: Common pitfalls or mistakes for this type of problem",
            "10. **modeling_tasks**: Overall classification of the modeling work needed",
            "11. **unresolved_questions**: What you still need to know",
            "12. **confidence**: Your overall confidence 0-1",
            "",
            "CRITICAL RULES:",
            "- Facts and data must come FROM the problem statement, not your knowledge",
            "- Assumptions must be EXPLICITLY marked as PROPOSED_ASSUMPTION",
            "- Never present assumptions as facts",
            "- Every subproblem must have a unique subproblem_id",
            "- Every evidence item must have a unique evidence_id",
            "- Every ambiguity must have a unique ambiguity_id",
            "- Do not select a model yet — that is a separate step",
            "- Do not fabricate numerical results",
        ])

        return "\n".join(parts)

    def _system_prompt(self) -> str:
        return (
            "You are an expert mathematical modeling competition analyst. "
            "You carefully read problem statements, identify what is fact vs. "
            "assumption, detect ambiguities, and decompose complex problems "
            "into well-defined subproblems. You never confuse facts with assumptions. "
            "You always provide structured, machine-readable output."
        )