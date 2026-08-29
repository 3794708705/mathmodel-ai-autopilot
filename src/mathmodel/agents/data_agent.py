"""MathModel AI — DataAgent.

Analyzes data structure and quality using Profile results.
Never fabricates numerical statistics — all numbers come from Python profiling.
"""

from __future__ import annotations

import logging
from typing import Type

from pydantic import BaseModel, Field

from mathmodel.agents.base import (
    AgentError,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from mathmodel.domain.analysis import ProblemAnalysis
from mathmodel.domain.data import (
    ColumnProfile,
    DataCleaningPlan,
    CleaningOperation,
    DataProfile,
)
from mathmodel.domain.state_helpers import load_analysis
from mathmodel.models.problem_state import ProblemState
from mathmodel.routing.profile import TaskProfile, TaskType
from mathmodel.routing.router import ModelRouter

logger = logging.getLogger(__name__)


class DataAnalysis(BaseModel):
    """Output of DataAgent: structured understanding of data."""

    data_summary: str = Field(..., min_length=1)
    table_relationships: list[str] = Field(default_factory=list)
    important_fields: list[str] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    quality_issues: list[str] = Field(default_factory=list)
    missing_data_strategy: str = ""
    outlier_strategy: str = ""
    feature_candidates: list[str] = Field(default_factory=list)
    target_candidates: list[str] = Field(default_factory=list)
    time_structure: str = ""
    spatial_structure: str = ""
    data_limitations: list[str] = Field(default_factory=list)
    modeling_implications: list[str] = Field(default_factory=list)
    cleaning_plan: DataCleaningPlan = Field(default_factory=DataCleaningPlan)


class DataAgent(BaseAgent):
    """Analyzes data structure and quality.

    Input: ProblemAnalysis + DataProfile
    Output: DataAnalysis with cleaning plan

    All numerical claims must reference ProfileResult or ExecutionRecord.
    Never fabricates statistics.
    """

    name = "DataAgent"
    role = "Data structure analysis and cleaning planning"
    input_schema = BaseModel
    output_schema = DataAnalysis
    capabilities = [
        "data_analysis",
        "quality_assessment",
        "cleaning_planning",
        "feature_identification",
    ]

    def __init__(self, router: ModelRouter):
        super().__init__()
        self._router = router

    async def run(self, state: ProblemState) -> AgentResult:
        result = self._start_result()

        try:
            analysis = load_analysis(state)
            if not analysis:
                return self._finish_result(
                    result, AgentStatus.FAILED,
                    errors=[AgentError(
                        message="No ProblemAnalysis found. Run ProblemAgent first.",
                        error_type="prerequisite",
                    )],
                )

            # Load profile from metadata
            profile_data = None
            if state.metadata_ and "data_profile" in state.metadata_:
                profile_data = DataProfile.model_validate(state.metadata_["data_profile"])

            prompt = self._build_prompt(analysis, profile_data)
            profile = TaskProfile.for_task_type(TaskType.DATA_ANALYSIS)

            output = await self._router.route_structured_generate(
                profile=profile,
                prompt=prompt,
                output_schema=DataAnalysis,
                system_prompt=self._system_prompt(),
            )

            if not isinstance(output, DataAnalysis):
                return self._finish_result(
                    result, AgentStatus.FAILED,
                    errors=[AgentError(
                        message=f"Expected DataAnalysis, got {type(output).__name__}",
                        error_type="schema",
                    )],
                )

            # Store in state
            if state.metadata_ is None:
                state.metadata_ = {}
            state.metadata_["data_analysis"] = output.model_dump()

            return self._finish_result(
                result, AgentStatus.COMPLETED,
                output=output.model_dump(),
            )

        except Exception as e:
            logger.exception("DataAgent failed")
            return self._finish_result(
                result, AgentStatus.FAILED,
                errors=[AgentError(message=str(e), error_type="runtime", recoverable=True)],
            )

    def validate_output(self, output: BaseModel) -> list[AgentError]:
        if not isinstance(output, DataAnalysis):
            return [AgentError(message=f"Expected DataAnalysis, got {type(output).__name__}", error_type="schema")]
        return []

    def _build_prompt(self, analysis: ProblemAnalysis, profile: DataProfile | None) -> str:
        parts = [
            "Analyze the data for a mathematical modeling competition problem.",
            f"## PROBLEM: {analysis.core_problem}",
        ]
        if profile:
            parts.append(f"## DATA PROFILE (computed by Python, NOT fabricated):")
            parts.append(f"Rows: {profile.row_count}, Columns: {profile.column_count}")
            parts.append(f"Missing values: {profile.total_missing}")
            parts.append(f"Duplicate rows: {profile.duplicate_rows}")
            for col in profile.columns:
                parts.append(f"  {col.name}: dtype={col.dtype}, missing={col.missing_rate:.1%}, "
                           f"unique={col.unique_count}, mean={col.mean}, median={col.median}")
        parts.extend([
            "## INSTRUCTIONS",
            "Provide: data_summary, quality_issues, feature_candidates, target_candidates, "
            "missing_data_strategy, outlier_strategy, data_limitations, modeling_implications.",
            "CRITICAL: All numerical values must come from the profile above. Do NOT fabricate numbers.",
        ])
        return "\n".join(parts)

    def _system_prompt(self) -> str:
        return "You are a data scientist analyzing competition datasets. You only report numbers from actual profiling, never fabricate statistics."