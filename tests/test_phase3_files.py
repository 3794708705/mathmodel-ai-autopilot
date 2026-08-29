"""Tests for Phase 3B: File Pipeline, Data Schemas, DataProfiler."""

import csv
import io
import os
import tempfile
from pathlib import Path

import pytest

from mathmodel.domain.data import (
    FileRecord,
    Dataset,
    DataTable,
    ColumnProfile,
    DataProfile,
    DataCleaningPlan,
    CleaningOperation,
    ParserStatus,
    SemanticType,
    DataLayer,
)
from mathmodel.files.service import FileService
from mathmodel.files.parsers import CsvParser, ExcelParser, TxtParser
from mathmodel.data import DataProfiler


# ═══════════════════════════════════════════════════════════════
# FileRecord
# ═══════════════════════════════════════════════════════════════

class TestFileRecord:
    def test_create_file_record(self):
        fr = FileRecord(
            original_name="test.csv",
            media_type="text/csv",
            extension=".csv",
            size_bytes=1024,
            sha256="abc123",
            storage_path="/tmp/test.csv",
        )
        assert fr.file_id.startswith("FILE-")
        assert fr.parser_status == ParserStatus.PENDING
        assert fr.is_original is True

    def test_safe_name_default(self):
        fr = FileRecord(original_name="test.csv")
        assert fr.safe_name == ""


# ═══════════════════════════════════════════════════════════════
# FileService
# ═══════════════════════════════════════════════════════════════

class TestFileService:
    def test_safe_filename(self):
        assert FileService._safe_filename("test.csv") == "test.csv"
        assert FileService._safe_filename("bad name.csv") == "bad_name.csv"
        assert FileService._safe_filename("path/traversal.csv") == "path_traversal.csv"
        assert FileService._safe_filename("../../../etc/passwd") in ("etc_passwd", "......_etc_passwd")

    def test_ingest_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = FileService(storage_dir=tmpdir)
            # Create a test CSV file
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")

            record = svc.ingest(csv_path)
            assert record.original_name == "test.csv"
            assert record.extension == ".csv"
            assert "csv" in record.media_type or record.extension == ".csv"
            assert record.sha256 != ""
            assert record.parser_name == "csv_parser"
            assert record.is_original is True

    def test_ingest_duplicate_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = FileService(storage_dir=tmpdir)
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b\n1,2\n")

            r1 = svc.ingest(csv_path)
            r2 = svc.ingest(csv_path)  # Same file
            assert r1.sha256 == r2.sha256  # Same hash

    def test_reject_unsafe_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = FileService(storage_dir=tmpdir)
            py_path = Path(tmpdir) / "test.py"
            py_path.write_text("print('hello')")
            with pytest.raises(ValueError, match="Unsafe"):
                svc.ingest(py_path)

    def test_reject_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = FileService(storage_dir=tmpdir)
            empty_path = Path(tmpdir) / "empty.csv"
            empty_path.write_text("")
            with pytest.raises(ValueError, match="Empty"):
                svc.ingest(empty_path)

    def test_reject_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = FileService(storage_dir=tmpdir)
            # Create file with path traversal in name
            bad_path = Path(tmpdir) / ".._etc_passwd.csv"
            bad_path.write_text("a\n1\n")
            with pytest.raises(ValueError, match="Unsafe"):
                svc.ingest(bad_path)


# ═══════════════════════════════════════════════════════════════
# CSV Parser
# ═══════════════════════════════════════════════════════════════

class TestCsvParser:
    def test_parse_simple_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("name,age,score\nAlice,30,85\nBob,25,92\n")

            fr = FileRecord(
                original_name="test.csv",
                storage_path=str(csv_path),
                media_type="text/csv",
                extension=".csv",
            )
            table, profile = CsvParser.parse(fr)

            assert table.row_count == 2
            assert table.column_count == 3
            assert len(profile.columns) == 3
            assert profile.columns[0].name == "name"
            assert profile.columns[1].name == "age"
            assert profile.columns[1].dtype == "int64"

    def test_parse_csv_with_missing_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b\n1,2\n,4\n3,\n")

            fr = FileRecord(
                original_name="test.csv",
                storage_path=str(csv_path),
                media_type="text/csv",
                extension=".csv",
            )
            table, profile = CsvParser.parse(fr)

            # Should have missing values
            missing = sum(c.missing_count for c in profile.columns)
            assert missing > 0

    def test_parse_csv_numeric_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("x\n1\n2\n3\n4\n5\n")

            fr = FileRecord(
                original_name="test.csv",
                storage_path=str(csv_path),
                media_type="text/csv",
                extension=".csv",
            )
            table, profile = CsvParser.parse(fr)

            col = profile.columns[0]
            assert col.min == 1.0
            assert col.max == 5.0
            assert col.mean == 3.0
            assert col.median == 3.0


# ═══════════════════════════════════════════════════════════════
# Data Schemas
# ═══════════════════════════════════════════════════════════════

class TestDataSchemas:
    def test_dataset_creation(self):
        ds = Dataset(
            name="Test Dataset",
            source_file_ids=["FILE-1"],
            layer=DataLayer.RAW,
        )
        assert ds.dataset_id.startswith("DS-")
        assert ds.version == 1

    def test_data_table_creation(self):
        tbl = DataTable(
            name="Sheet1",
            row_count=100,
            column_count=5,
            columns=["COL-1", "COL-2"],
        )
        assert tbl.table_id.startswith("TBL-")

    def test_column_profile(self):
        col = ColumnProfile(
            name="age",
            dtype="int64",
            semantic_type=SemanticType.NUMERIC,
            min=18.0,
            max=65.0,
            mean=35.0,
            median=34.0,
        )
        assert col.column_id.startswith("COL-")
        assert col.semantic_type == SemanticType.NUMERIC

    def test_cleaning_plan(self):
        op = CleaningOperation(
            target="COL-1",
            operation="fill_mean",
            reason="Missing values in age column",
            risk="low",
            reversible=True,
        )
        plan = DataCleaningPlan(
            dataset_id="DS-1",
            operations=[op],
        )
        assert len(plan.operations) == 1
        assert plan.operations[0].operation == "fill_mean"


# ═══════════════════════════════════════════════════════════════
# DataProfiler
# ═══════════════════════════════════════════════════════════════

class TestDataProfiler:
    def test_profile_empty(self):
        table = DataTable(name="empty", row_count=0, column_count=0)
        profile = DataProfiler.profile_table(table, [], [])
        assert profile.row_count == 0

    def test_profile_numeric(self):
        table = DataTable(name="test", row_count=3, column_count=2)
        rows = [[1, 10], [2, 20], [3, 30]]
        headers = ["x", "y"]
        profile = DataProfiler.profile_table(table, rows, headers)

        assert len(profile.columns) == 2
        assert profile.columns[0].dtype == "int64"
        assert profile.columns[0].mean == 2.0

    def test_profile_mixed_types(self):
        table = DataTable(name="test", row_count=3, column_count=2)
        rows = [["Alice", "30"], ["Bob", "25"], ["Charlie", "35"]]
        headers = ["name", "age"]
        profile = DataProfiler.profile_table(table, rows, headers)

        # name should be text/categorical, age should be numeric
        assert profile.columns[0].semantic_type in (SemanticType.TEXT, SemanticType.CATEGORICAL, SemanticType.IDENTIFIER)
        assert profile.columns[1].dtype == "int64"

    def test_profile_outliers(self):
        table = DataTable(name="test", row_count=10, column_count=1)
        rows = [[1], [2], [2], [3], [3], [3], [4], [4], [5], [100]]
        headers = ["x"]
        profile = DataProfiler.profile_table(table, rows, headers)

        # 100 should be an outlier
        assert len(profile.outlier_candidates) > 0

    def test_profile_correlation(self):
        table = DataTable(name="test", row_count=5, column_count=2)
        rows = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]
        headers = ["x", "y"]
        profile = DataProfiler.profile_table(table, rows, headers)

        # Perfect correlation
        assert len(profile.correlation_candidates) > 0
        assert abs(profile.correlation_candidates[0]["correlation"]) == 1.0