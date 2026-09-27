"""MathModel AI — ModelJury agent.

Scores and ranks eligible candidate models.
LLM provides dimension scores; Python computes weighted totals.
"""

from __future__ import annotations

import asyncio

import logging
from typing import Type

from pydantic import BaseModel, Field

from mathmodel.agents.base import (
    AgentError,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from mathmodel.domain.candidates import ModelCandidate
from mathmodel.domain.eligibility import EligibilityResult
from mathmodel.domain.jury import (
    DEFAULT_JURY_WEIGHTS,
    JuryDimension,
    JuryScore,
    ModelJuryResult,
    SelectionOverride,
    compute_jury_scores,
)
from mathmodel.domain.state_helpers import (
    load_candidates,
    load_eligibility_results,
    record_revision,
    store_jury_result,
)
from mathmodel.models.problem_state import ProblemState, ProblemStateStage
from mathmodel.routing.profile import TaskProfile, TaskType
from mathmodel.routing.router import ModelRouter

logger = logging.getLogger(__name__)


class JuryScoreOutput(BaseModel):
    """LLM output: raw scores per dimension."""
    raw_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores 0-100 per JuryDimension",
    )
    reasoning: dict[str, str] = Field(
        default_factory=dict,
        description="One-sentence reasoning per dimension",
    )


class ModelJury(BaseAgent):
    """Scores and ranks eligible candidate models.

    Deterministic: LLM provides raw dimension scores, Python computes
    weighted totals. The LLM never computes the final score.
    """

    name = "ModelJury"
    role = "Candidate model evaluation and selection"
    input_schema = BaseModel
    output_schema = ModelJuryResult
    capabilities = [
        "model_scoring",
        "model_ranking",
        "model_selection",
    ]

    def __init__(
        self,
        router: ModelRouter,
        weights: dict[JuryDimension, float] | None = None,
    ):
        super().__init__()
        self._router = router
        self._weights = weights or DEFAULT_JURY_WEIGHTS

    async def run(self, state: ProblemState) -> AgentResult:
        result = self._start_result()

        try:
            candidates = load_candidates(state)
            if not candidates:
                return self._finish_result(
                    result,
                    AgentStatus.FAILED,
                    errors=[AgentError(
                        message="No candidates found in state",
                        error_type="prerequisite",
                    )],
                )

            eligibility_results = load_eligibility_results(state)
            eligible_ids = {
                r.candidate_id for r in eligibility_results if r.eligible
            }

            # Score each eligible candidate
            scores: list[JuryScore] = []
            for candidate in candidates:
                if candidate.candidate_id not in eligible_ids:
                    continue

                score = await self._score_candidate(candidate)
                scores.append(score)

            if not scores:
                return self._finish_result(
                    result,
                    AgentStatus.FAILED,
                    errors=[AgentError(
                        message="No eligible candidates to score",
                        error_type="prerequisite",
                    )],
                )

            # Rank by total_score (descending)
            scores.sort(key=lambda s: s.total_score, reverse=True)
            ranking = [s.candidate_id for s in scores]

            # Select primary and backup
            selected = ranking[0] if ranking else None
            backup = ranking[1] if len(ranking) > 1 else None

            # Build jury result
            jury_result = ModelJuryResult(
                candidate_scores=scores,
                ranking=ranking,
                selected_model=selected,
                backup_model=backup,
                rejected_models=[
                    c.candidate_id for c in candidates
                    if c.candidate_id not in eligible_ids
                ],
                decision_reason=(
                    f"Selected {selected} (score: {scores[0].total_score}). "
                    f"Backup: {backup or 'none'}. "
                    f"Rejected (ineligible): {len(candidates) - len(scores)}"
                ),
                risks=[],
                confidence=0.8,
            )

            # Store in ProblemState
            store_jury_result(state, jury_result)
            record_revision(
                state,
                self.name,
                ProblemStateStage.SELECT,
                f"Selected model: {selected}",
            )

            state.current_stage = ProblemStateStage.SELECT

            return self._finish_result(
                result,
                AgentStatus.COMPLETED,
                output=jury_result.model_dump(),
            )

        except Exception as e:
            logger.exception("ModelJury failed")
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
        if not isinstance(output, ModelJuryResult):
            return [AgentError(
                message=f"Expected ModelJuryResult, got {type(output).__name__}",
                error_type="schema",
            )]

        errors = []
        if output.selected_model and output.selected_model not in output.ranking:
            errors.append(AgentError(
                message="selected_model not in ranking",
                error_type="validation",
            ))
        if output.backup_model and output.backup_model not in output.ranking:
            errors.append(AgentError(
                message="backup_model not in ranking",
                error_type="validation",
            ))
        if output.selected_model == output.backup_model:
            errors.append(AgentError(
                message="selected_model and backup_model must differ",
                error_type="validation",
            ))

        return errors

    async def _score_candidate(
        self, candidate: ModelCandidate
    ) -> JuryScore:
        """Score a single candidate using LLM + deterministic computation."""
        prompt = self._build_score_prompt(candidate)
        profile = TaskProfile.for_task_type(TaskType.MODEL_JURY)

        # A transient gateway failure on ONE candidate must not discard the whole
        # run. Measured on this project: a single `httpx2.ReadError` surfaced as
        # `openai.APIConnectionError` from this call, propagated out of `run`, and
        # killed a run that had already completed intake, understanding, evidence
        # extraction and candidate exploration -- the run ended `blocked` with
        # "model selection failed" and every earlier stage's work was lost. The
        # error is a transport blip, not a judgement about the candidate, so retry
        # briefly before giving up.
        llm_output = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                llm_output = await self._router.route_structured_generate(
                    profile=profile,
                    prompt=prompt,
                    output_schema=JuryScoreOutput,
                    system_prompt=self._system_prompt(),
                )
                break
            except Exception as exc:  # noqa: BLE001 - transport errors vary
                last_error = exc
                if attempt < 2:
                    logger.warning(
                        "ModelJury: scoring %s failed on attempt %d (%s); retrying",
                        candidate.candidate_id,
                        attempt + 1,
                        exc,
                    )
                    await asyncio.sleep(2.0 * (attempt + 1))
        if llm_output is None:
            raise RuntimeError(
                f"ModelJury could not score {candidate.candidate_id} after 3 "
                f"attempts: {last_error}"
            ) from last_error

        if not isinstance(llm_output, JuryScoreOutput):
            raise ValueError(
                f"Expected JuryScoreOutput, got {type(llm_output).__name__}"
            )

        # Convert string keys to enum keys
        raw_scores: dict[JuryDimension, float] = {}
        reasoning: dict[JuryDimension, str] = {}
        missing_dimensions: list[str] = []

        for dim in JuryDimension:
            if dim.value in llm_output.raw_scores:
                raw_scores[dim] = llm_output.raw_scores[dim.value]
                reasoning[dim] = llm_output.reasoning.get(dim.value, "")
            else:
                missing_dimensions.append(dim.value)
                raw_scores[dim] = 50.0
                reasoning[dim] = ""

        if missing_dimensions:
            logger.warning(
                "ModelJury: LLM omitted dimensions: %s. Defaulting to 50.0.",
                missing_dimensions,
            )

        # Compute weighted scores deterministically
        return compute_jury_scores(
            candidate_id=candidate.candidate_id,
            raw_scores=raw_scores,
            reasoning=reasoning,
            weights=self._weights,
        )

    def _build_score_prompt(self, candidate: ModelCandidate) -> str:
        return "\n".join([
            "Score this candidate mathematical model for a competition problem.",
            "",
            f"## CANDIDATE: {candidate.name}",
            f"Model family: {candidate.model_family.value}",
            f"Summary: {candidate.summary}",
            f"Mathematical structure: {candidate.mathematical_structure}",
            f"Strengths: {'; '.join(candidate.strengths)}",
            f"Weaknesses: {'; '.join(candidate.weaknesses)}",
            f"Implementation plan: {candidate.implementation_plan}",
            f"Validation plan: {candidate.validation_plan}",
            "",
            "## SCORING",
            "Score each dimension 0-100:",
            "- problem_fit: How well does this model match the problem?",
            "- data_fit: How well does this model use available data?",
            "- mathematical_validity: Is the math sound?",
            "- explainability: Can results be explained to judges?",
            "- validation_potential: Can this model be validated?",
            "- innovation: Is this approach creative?",
            "- competition_feasibility: Can this be completed in competition time?",
            "- computational_efficiency: Is the computation practical?",
            "",
            "Provide raw_scores (0-100) and one-sentence reasoning per dimension.",
        ])

    def _system_prompt(self) -> str:
        return (
            "You are a mathematical modeling competition judge. "
            "You evaluate candidate models on fit, validity, feasibility, "
            "and innovation. You provide honest, calibrated scores. "
            "You do not inflate scores — a 90 means truly excellent."
        )