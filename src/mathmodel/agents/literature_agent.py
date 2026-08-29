"""MathModel AI — LiteratureAgent skeleton.

Plans literature search queries based on problem analysis and
candidate models. Does NOT fabricate papers, DOIs, or citations.
All queries remain SEARCH_PENDING until a real search tool exists.
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
from mathmodel.domain.analysis import ProblemAnalysis
from mathmodel.domain.candidates import ModelCandidate
from mathmodel.domain.literature import (
    LiteratureQuery,
    LiteratureQueryPlan,
    LiteratureQueryStatus,
)
from mathmodel.domain.state_helpers import (
    load_analysis,
    load_candidates,
    store_literature_plan,
)
from mathmodel.models.problem_state import ProblemState

logger = logging.getLogger(__name__)


class LiteratureAgent(BaseAgent):
    """Plans literature search queries. Skeleton implementation.

    Creates a LiteratureQueryPlan with targeted queries based on
    the problem analysis and candidate models.

    NO papers, DOIs, authors, or citations are fabricated.
    All queries are SEARCH_PENDING until a real search tool is available.
    """

    name = "LiteratureAgent"
    role = "Literature search planning"
    input_schema = BaseModel
    output_schema = LiteratureQueryPlan
    capabilities = [
        "literature_planning",
        "query_formulation",
    ]

    async def run(self, state: ProblemState) -> AgentResult:
        result = self._start_result()

        try:
            analysis = load_analysis(state)
            candidates = load_candidates(state)

            if not analysis:
                return self._finish_result(
                    result,
                    AgentStatus.FAILED,
                    errors=[AgentError(
                        message="No ProblemAnalysis found. Run ProblemAgent first.",
                        error_type="prerequisite",
                    )],
                )

            # Build queries based on analysis and candidates
            queries = self._build_queries(analysis, candidates)

            plan = LiteratureQueryPlan(
                queries=queries,
            )

            # Store in ProblemState
            store_literature_plan(state, plan)

            return self._finish_result(
                result,
                AgentStatus.COMPLETED,
                output=plan.model_dump(),
            )

        except Exception as e:
            logger.exception("LiteratureAgent failed")
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
        if not isinstance(output, LiteratureQueryPlan):
            return [AgentError(
                message=f"Expected LiteratureQueryPlan, got {type(output).__name__}",
                error_type="schema",
            )]

        # Verify no fabricated results
        for q in output.queries:
            if q.status != LiteratureQueryStatus.SEARCH_PENDING:
                return [AgentError(
                    message=f"Query {q.query_id} has status {q.status.value} "
                            f"but no real search was performed",
                    error_type="fabrication",
                )]

        return []

    def _build_queries(
        self,
        analysis: ProblemAnalysis,
        candidates: list[ModelCandidate],
    ) -> list[LiteratureQuery]:
        """Build literature queries from analysis and candidates.

        These are search plans — NOT fabricated results.
        """
        queries = []

        # Problem-level queries
        if analysis.core_problem:
            queries.append(LiteratureQuery(
                purpose="Find related work on the core problem",
                keywords=self._extract_keywords(analysis.core_problem),
                target_problem=analysis.core_problem[:200],
            ))

        # Method-level queries per candidate
        for candidate in candidates:
            queries.append(LiteratureQuery(
                purpose=f"Find literature supporting {candidate.name}",
                keywords=[
                    candidate.model_family.value.replace("_", " "),
                    *candidate.name.split(),
                ],
                target_method=candidate.model_family.value,
                candidate_ids=[candidate.candidate_id],
                priority=2,
            ))

        # Subproblem-specific queries
        for sp in analysis.subproblems:
            if sp.task_types:
                queries.append(LiteratureQuery(
                    purpose=f"Find methods for {sp.normalized_goal[:100]}",
                    keywords=[
                        sp.normalized_goal[:50],
                        *(t.value for t in sp.task_types),
                    ],
                    target_application=sp.normalized_goal[:200],
                    priority=1,
                ))

        return queries

    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
        """Extract simple keywords from text."""
        # Simple extraction: take significant words
        words = text.lower().split()
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "of", "in",
            "to", "for", "and", "or", "on", "at", "by", "with", "from",
            "this", "that", "it", "be", "as", "has", "have", "do", "does",
            "will", "would", "can", "could", "should", "may", "might",
        }
        keywords = [
            w.strip(".,;:!?()[]{}") for w in words
            if w.strip(".,;:!?()[]{}") not in stop_words
            and len(w.strip(".,;:!?()[]{}")) > 3
        ]
        return keywords[:max_keywords]