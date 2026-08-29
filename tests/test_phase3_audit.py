"""Phase 3 audit adversarial tests.

Covers: path traversal, malformed parsers, data truth, sandbox boundary,
execution truth, retry exhaustion, state round-trip.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mathmodel.data import DataProfiler
from mathmodel.domain.data import (
    DataTable,
    DataProfile,
    ColumnProfile,
    SemanticType,
    Dataset,
    DataLayer,
    FileRecord,
    ParserStatus,
)
from mathmodel.files.parsers import CsvParser, TxtParser
from mathmodel.files.service import FileService
from mathmodel.sandbox.backend import (
    ExecutionStatus,
    ExecutionRecord,
    LocalTestSandboxBackend,
    SandboxLimits,
)
from mathmodel.sandbox.executor import SandboxExecutor


def run_async(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════
# FileService — Path Traversal
# ═══════════════════════════════════════════════════════════════

class TestPathTraversal:
    def test_directory_traversal_in_path(self):
        """Path traversal in directory components should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a nested dir with a safe file
            nested = Path(tmpdir) / "subdir"
            nested.mkdir()
            safe_file = nested / "safe.csv"
            safe_file.write_text("a\n1\n")

            svc = FileService(storage_dir=tmpdir)

            # Try to access via traversal path
            traversal_path = Path(tmpdir) / ".." / Path(tmpdir).name / "subdir" / "safe.csv"
            # This path resolves correctly but contains ".."
            if traversal_path.exists():
                with pytest.raises(ValueError, match="traversal"):
                    svc.ingest(str(traversal_path))

    def test_safe_filename_prevents_escape(self):
        """Safe filename should not contain path separators."""
        # After safe_filename, the result is used as a single filename
        safe = FileService._safe_filename("../../../etc/passwd")
        assert "/" not in safe
        assert "\\" not in safe
        assert not safe.startswith(".")

    def test_original_files_not_overwritten(self):
        """Two files with same name should not overwrite each other."""
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = FileService(storage_dir=tmpdir)
            csv_path1 = Path(tmpdir) / "data.csv"
            csv_path1.write_text("a\n1\n")
            csv_path2 = Path(tmpdir) / "data.csv"
            csv_path2.write_text("a\n2\n")

            r1 = svc.ingest(csv_path1)
            r2 = svc.ingest(csv_path2)

            # Both should exist with different paths
            assert r1.storage_path != r2.storage_path
            assert Path(r1.storage_path).exists()
            assert Path(r2.storage_path).exists()


# ═══════════════════════════════════════════════════════════════
# CSV Parser — Malformed Input
# ═══════════════════════════════════════════════════════════════

class TestCsvParserAdversarial:
    def test_empty_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            csv_path.write_text("")
            fr = FileRecord(
                original_name="empty.csv",
                storage_path=str(csv_path),
                media_type="text/csv",
                extension=".csv",
            )
            with pytest.raises(ValueError, match="empty"):
                CsvParser.parse(fr)

    def test_mixed_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "mixed.csv"
            csv_path.write_text("col\n1\ntwo\n3.5\n")
            fr = FileRecord(
                original_name="mixed.csv",
                storage_path=str(csv_path),
                media_type="text/csv",
                extension=".csv",
            )
            table, profile = CsvParser.parse(fr)
            # Should not crash on mixed types
            assert table.row_count == 3
            assert len(profile.columns) == 1

    def test_all_null_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "nulls.csv"
            csv_path.write_text("a,b\n1,\n2,\n3,\n")
            fr = FileRecord(
                original_name="nulls.csv",
                storage_path=str(csv_path),
                media_type="text/csv",
                extension=".csv",
            )
            table, profile = CsvParser.parse(fr)
            # Column b should be all missing
            col_b = profile.columns[1]
            assert col_b.missing_count == 3
            assert col_b.missing_rate == 1.0

    def test_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "dup.csv"
            csv_path.write_text("a,a\n1,2\n")
            fr = FileRecord(
                original_name="dup.csv",
                storage_path=str(csv_path),
                media_type="text/csv",
                extension=".csv",
            )
            table, profile = CsvParser.parse(fr)
            # Should handle duplicate headers (both columns named "a")
            assert table.column_count == 2
            assert profile.columns[0].name == "a"
            assert profile.columns[1].name == "a"


# ═══════════════════════════════════════════════════════════════
# DataProfiler — Numerical Truth
# ═══════════════════════════════════════════════════════════════

class TestDataProfilerAdversarial:
    def test_constant_column(self):
        table = DataTable(name="test", row_count=3, column_count=1)
        rows = [[5], [5], [5]]
        headers = ["x"]
        profile = DataProfiler.profile_table(table, rows, headers)

        col = profile.columns[0]
        assert col.mean == 5.0
        assert col.std == 0.0
        assert col.min == col.max == 5.0

    def test_single_row(self):
        table = DataTable(name="test", row_count=1, column_count=1)
        rows = [[42]]
        headers = ["x"]
        profile = DataProfiler.profile_table(table, rows, headers)

        col = profile.columns[0]
        assert col.mean == 42.0
        assert col.std == 0.0  # Single row

    def test_outlier_uses_correct_column(self):
        """Verify outlier detection uses the correct column's values."""
        table = DataTable(name="test", row_count=5, column_count=2)
        rows = [[1, 100], [2, 200], [3, 300], [4, 400], [100, 500]]
        headers = ["x", "y"]
        profile = DataProfiler.profile_table(table, rows, headers)

        # Column x: 1,2,3,4,100 → 100 is outlier
        # Column y: 100,200,300,400,500 → no outlier
        outliers = profile.outlier_candidates
        # Should have at least one outlier (from column x)
        assert len(outliers) >= 1
        # The outlier should be in column x, not y
        outlier_cols = [o["column"] for o in outliers]
        assert "x" in outlier_cols

    def test_empty_dataset(self):
        table = DataTable(name="empty", row_count=0, column_count=0)
        profile = DataProfiler.profile_table(table, [], [])
        assert profile.row_count == 0
        assert profile.column_count == 0


# ═══════════════════════════════════════════════════════════════
# Sandbox — Execution Truth
# ═══════════════════════════════════════════════════════════════

class TestSandboxExecutionTruth:
    def test_success_has_exit_code_zero(self):
        record = run_async(SandboxExecutor().execute("print('ok')"))
        assert record.status == ExecutionStatus.SUCCESS
        assert record.exit_code == 0

    def test_failure_has_nonzero_exit(self):
        record = run_async(SandboxExecutor().execute("x = 1/0"))
        assert record.status == ExecutionStatus.FAILED
        assert record.exit_code != 0

    def test_timeout_has_timed_out_flag(self):
        executor = SandboxExecutor(limits=SandboxLimits(timeout_seconds=1))
        record = run_async(executor.execute("import time; time.sleep(10)"))
        assert record.status == ExecutionStatus.TIMED_OUT
        assert record.timed_out is True

    def test_no_execution_no_success(self):
        """Never create a SUCCESS record without actual execution."""
        record = ExecutionRecord(status=ExecutionStatus.PENDING)
        assert record.status != ExecutionStatus.SUCCESS
        assert record.exit_code == -1

    def test_mock_flag_on_local_backend(self):
        record = run_async(SandboxExecutor().execute("print('test')"))
        # Local backend: real execution, NOT mock, but NOT production safe
        assert record.is_mock is False
        assert record.execution_real is True
        assert record.production_safe is False

    def test_stdout_captured(self):
        record = run_async(SandboxExecutor().execute("print('hello sandbox')"))
        assert "hello sandbox" in record.stdout

    def test_stderr_captured(self):
        record = run_async(SandboxExecutor().execute("import sys; print('err', file=sys.stderr)"))
        assert "err" in record.stderr

    def test_artifact_collected(self):
        code = "with open('result.txt', 'w') as f: f.write('data')"
        record = run_async(SandboxExecutor().execute(code))
        assert "result.txt" in record.artifacts

    def test_unsafe_backend_label(self):
        backend = LocalTestSandboxBackend()
        assert backend.UNSAFE_FOR_PRODUCTION is True
        assert backend.production_safe is False

    def test_security_blocks_network(self):
        code = "import urllib.request; urllib.request.urlopen('http://example.com')"
        record = run_async(SandboxExecutor().execute(code))
        assert record.status == ExecutionStatus.SECURITY_VIOLATION

    def test_security_blocks_subprocess(self):
        code = "import subprocess; subprocess.run(['echo', 'test'])"
        record = run_async(SandboxExecutor().execute(code))
        assert record.status == ExecutionStatus.SECURITY_VIOLATION


# ═══════════════════════════════════════════════════════════════
# State Round-trip
# ═══════════════════════════════════════════════════════════════

class TestStateRoundTrip:
    def test_dataset_round_trip(self):
        ds = Dataset(
            name="Test",
            source_file_ids=["FILE-1", "FILE-2"],
            layer=DataLayer.RAW,
            version=3,
            parent_dataset_id="DS-PARENT",
        )
        data = ds.model_dump()
        restored = Dataset.model_validate(data)
        assert restored.name == "Test"
        assert restored.version == 3
        assert restored.parent_dataset_id == "DS-PARENT"
        assert restored.layer == DataLayer.RAW

    def test_malformed_dataset_rejected(self):
        with pytest.raises(Exception):
            Dataset.model_validate({"name": 123, "layer": "not_a_layer"})

    def test_execution_record_round_trip(self):
        from dataclasses import asdict
        record = ExecutionRecord(
            code_hash="abc123",
            backend="test",
            status=ExecutionStatus.SUCCESS,
            exit_code=0,
            stdout="hello",
            stderr="",
            runtime_seconds=1.5,
            artifacts=["output.txt"],
        )
        d = asdict(record)
        assert d["status"] == "success"
        assert d["exit_code"] == 0
        assert "output.txt" in d["artifacts"]

    def test_data_profile_round_trip(self):
        profile = DataProfile(
            dataset_id="DS-1",
            table_id="TBL-1",
            row_count=100,
            column_count=3,
            columns=[
                ColumnProfile(
                    name="sales",
                    dtype="float64",
                    semantic_type=SemanticType.NUMERIC,
                    mean=50.0,
                    median=45.0,
                    min=10.0,
                    max=100.0,
                    std=15.0,
                    missing_count=2,
                    missing_rate=0.02,
                ),
            ],
            total_missing=2,
            duplicate_rows=0,
            outlier_candidates=[{"column": "sales", "method": "IQR", "count": 3}],
            correlation_candidates=[{"col_a": "sales", "col_b": "price", "correlation": 0.85}],
        )
        data = profile.model_dump()
        restored = DataProfile.model_validate(data)
        assert restored.row_count == 100
        assert restored.columns[0].mean == 50.0
        assert len(restored.outlier_candidates) == 1
        assert len(restored.correlation_candidates) == 1