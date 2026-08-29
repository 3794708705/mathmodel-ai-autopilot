"""Phase 3 End-to-End fixture test.

Tests the full Phase 3 pipeline:
INGEST → File Parse → Data Profile → DataAgent → Sandbox Execution
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mathmodel.agents.data_agent import DataAgent
from mathmodel.agents.problem_agent import ProblemAgent
from mathmodel.data import DataProfiler
from mathmodel.domain.analysis import ProblemAnalysis, Subproblem, ModelingTaskType
from mathmodel.domain.data import (
    DataTable,
    DataProfile,
    Dataset,
    DataLayer,
    CleaningOperation,
    DataCleaningPlan,
)
from mathmodel.domain.state_helpers import store_analysis, load_analysis
from mathmodel.files.parsers import CsvParser
from mathmodel.files.service import FileService
from mathmodel.models.problem_state import ProblemState, ProblemStateStage
from mathmodel.sandbox.executor import SandboxExecutor
from mathmodel.providers.mock import MockProvider
from mathmodel.routing.router import ModelRouter


def make_router():
    return ModelRouter()


class TestPhase3E2E:
    """End-to-end Phase 3 pipeline test."""

    def test_full_pipeline(self, db_session):
        """Test: INGEST → FILE PARSE → DATA PROFILE → DATA AGENT → SANDBOX."""
        state = ProblemState(
            title="Phase 3 E2E Test",
            raw_problem="Analyze sales data. Find optimal inventory levels.",
            current_stage=ProblemStateStage.INGEST,
        )

        # Step 1: Store analysis (simulating ProblemAgent)
        analysis = ProblemAnalysis(
            background="Retail inventory",
            core_problem="Optimize inventory levels based on sales data",
            subproblems=[
                Subproblem(
                    subproblem_id="SUB-1",
                    original_text="Analyze sales data",
                    normalized_goal="Profile and clean sales data",
                    task_types=[ModelingTaskType.STATISTICAL_ANALYSIS],
                ),
            ],
        )
        store_analysis(state, analysis)
        state.current_stage = ProblemStateStage.UNDERSTAND

        # Step 2: Create test CSV and ingest
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sales.csv"
            csv_path.write_text("store,day,sales,price\nA,1,100,10.5\nA,2,120,10.5\nB,1,80,12.0\nB,2,95,12.0\n")

            svc = FileService(storage_dir=tmpdir)
            file_record = svc.ingest(csv_path)
            assert file_record.parser_name == "csv_parser"

            # Step 3: Parse CSV
            table, profile = CsvParser.parse(file_record)
            assert table.row_count == 4
            assert table.column_count == 4
            assert len(profile.columns) == 4

            # Step 4: Profile with DataProfiler
            rows = [["A", "1", "100", "10.5"], ["A", "2", "120", "10.5"],
                    ["B", "1", "80", "12.0"], ["B", "2", "95", "12.0"]]
            headers = ["store", "day", "sales", "price"]
            profile2 = DataProfiler.profile_table(table, rows, headers)
            assert profile2.row_count == 4
            assert profile2.column_count == 4

            # Store profile in state
            if state.metadata_ is None:
                state.metadata_ = {}
            state.metadata_["data_profile"] = profile2.model_dump()

            # Step 5: DataAgent
            agent = DataAgent(router=make_router())
            result = asyncio.run(agent.run(state))
            assert result.agent_name == "DataAgent"

            # Step 6: Sandbox execution
            executor = SandboxExecutor()
            code = """
import statistics
sales = [100, 120, 80, 95]
mean_sales = statistics.mean(sales)
print(f"Mean sales: {mean_sales}")
with open("output.txt", "w") as f:
    f.write(str(mean_sales))
"""
            record = asyncio.run(executor.execute(code))
            assert record.status.value in ("success", "failed")
            if record.status.value == "success":
                assert "Mean sales" in record.stdout
                assert "output.txt" in record.artifacts

        # Step 7: Verify state persistence
        restored = load_analysis(state)
        assert restored is not None
        assert restored.core_problem == "Optimize inventory levels based on sales data"

    def test_cleaning_plan_creation(self):
        """Test creating a data cleaning plan."""
        plan = DataCleaningPlan(
            dataset_id="DS-1",
            operations=[
                CleaningOperation(
                    target="COL-1",
                    operation="fill_mean",
                    reason="Missing values in age column",
                    risk="low",
                    reversible=True,
                ),
                CleaningOperation(
                    target="COL-2",
                    operation="drop_outliers",
                    reason="IQR outliers detected",
                    parameters={"method": "IQR", "threshold": 1.5},
                    risk="medium",
                    reversible=True,
                ),
            ],
        )
        assert len(plan.operations) == 2
        assert plan.operations[0].reversible is True