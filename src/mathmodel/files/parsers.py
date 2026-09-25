"""MathModel AI — File parsers.

CSV, Excel, PDF, and TXT parsers.
Each parser produces structured metadata and profile data.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any, Optional

from mathmodel.domain.data import (
    ColumnProfile,
    DataProfile,
    DataTable,
    FileRecord,
    ParserStatus,
    SemanticType,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CSV Parser
# ═══════════════════════════════════════════════════════════════

class CsvParser:
    """Parser for CSV files. Produces DataTable and DataProfile."""

    @staticmethod
    def parse(file_record: FileRecord) -> tuple[DataTable, DataProfile]:
        """Parse a CSV file and return table + profile."""
        path = Path(file_record.storage_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Try to detect encoding and delimiter
        content = path.read_bytes()

        # Try UTF-8 first
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        # Sniff delimiter
        sample = text[:8192]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel()

        reader = csv.reader(io.StringIO(text), dialect)
        rows = list(reader)

        if not rows:
            raise ValueError("CSV file is empty")

        # First row is header
        headers = rows[0]
        data_rows = rows[1:]

        # Build table
        table = DataTable(
            name=file_record.original_name,
            source=str(path),
            row_count=len(data_rows),
            column_count=len(headers),
        )

        # Build column profiles
        columns = []
        for idx, header in enumerate(headers):
            col_values = [
                row[idx] if idx < len(row) else None
                for row in data_rows
            ]

            col = CsvParser._profile_column(
                idx, header, col_values
            )
            columns.append(col)
            table.columns.append(col.column_id)

        table.column_count = len(columns)

        profile = DataProfile(
            dataset_id="",
            table_id=table.table_id,
            row_count=table.row_count,
            column_count=table.column_count,
            columns=columns,
            total_missing=sum(c.missing_count for c in columns),
            duplicate_rows=CsvParser._count_duplicates(data_rows),
        )

        return table, profile

    @staticmethod
    def _cell_to_str(row: list[Any], idx: int) -> Optional[str]:
        """Render one Excel cell as a string for profiling (None stays None)."""
        if idx >= len(row):
            return None
        value = row[idx]
        if value is None:
            return None
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    @staticmethod
    def _profile_column(
        idx: int, name: str, values: list[Optional[str]]
    ) -> ColumnProfile:
        """Profile a single column from CSV values."""
        non_null = [v for v in values if v is not None and v.strip() != ""]
        missing_count = len(values) - len(non_null)
        missing_rate = missing_count / len(values) if values else 0.0

        # Detect dtype and semantic type
        dtype, semantic_type = CsvParser._detect_type(non_null)

        col = ColumnProfile(
            name=name,
            original_name=name,
            dtype=dtype,
            semantic_type=semantic_type,
            missing_count=missing_count,
            missing_rate=round(missing_rate, 4),
            unique_count=len(set(non_null)) if non_null else 0,
            example_values=non_null[:5],
        )

        # Numeric stats
        if dtype in ("int64", "float64"):
            nums = CsvParser._to_numeric(non_null)
            if nums:
                col.min = min(nums)
                col.max = max(nums)
                col.mean = sum(nums) / len(nums)
                col.std = CsvParser._std(nums, col.mean)
                sorted_nums = sorted(nums)
                col.median = sorted_nums[len(sorted_nums) // 2]
                col.quantiles = {
                    "q25": sorted_nums[len(sorted_nums) // 4],
                    "q50": col.median,
                    "q75": sorted_nums[3 * len(sorted_nums) // 4],
                }

        return col

    @staticmethod
    def _detect_type(values: list[str]) -> tuple[str, SemanticType]:
        """Detect dtype and semantic type from string values."""
        if not values:
            return "object", SemanticType.UNKNOWN

        # Try numeric
        int_count = 0
        float_count = 0
        for v in values[:100]:
            try:
                int(v)
                int_count += 1
            except (ValueError, TypeError):
                try:
                    float(v)
                    float_count += 1
                except (ValueError, TypeError):
                    pass

        sample_size = min(100, len(values))
        if int_count == sample_size:
            return "int64", SemanticType.NUMERIC
        if int_count + float_count == sample_size:
            return "float64", SemanticType.NUMERIC

        # Check uniqueness ratio for identifier
        unique_ratio = len(set(values)) / len(values) if values else 0
        if unique_ratio > 0.9 and len(values) > 10:
            return "object", SemanticType.IDENTIFIER

        # Check cardinality for categorical
        if len(set(values)) < min(20, len(values) * 0.3):
            return "object", SemanticType.CATEGORICAL

        return "object", SemanticType.TEXT

    @staticmethod
    def _to_numeric(values: list[str]) -> list[float]:
        """Convert string values to floats, skipping non-numeric."""
        result = []
        for v in values:
            try:
                result.append(float(v))
            except (ValueError, TypeError):
                pass
        return result

    @staticmethod
    def _std(values: list[float], mean: float) -> float:
        if len(values) < 2:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5

    @staticmethod
    def _count_duplicates(rows: list[list[str]]) -> int:
        seen = set()
        dupes = 0
        for row in rows:
            key = tuple(row)
            if key in seen:
                dupes += 1
            else:
                seen.add(key)
        return dupes


# ═══════════════════════════════════════════════════════════════
# Excel Parser
# ═══════════════════════════════════════════════════════════════

class ExcelParser:
    """Parser for Excel (.xlsx, .xls) files.

    Reads real cell values — headers, rows, and per-column profiles.
    A sheet is never summarized from its filename or dimensions alone.
    """

    @staticmethod
    def read_sheets(file_record: FileRecord) -> dict[str, dict[str, Any]]:
        """Read every sheet with real cell values.

        Returns {sheet_name: {"headers": [...], "rows": [[...], ...]}}.
        """
        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl is required for Excel parsing")

        path = Path(file_record.storage_path)
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

        sheets: dict[str, dict[str, Any]] = {}
        try:
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                raw_rows = list(ws.iter_rows(values_only=True))
                if not raw_rows:
                    continue
                headers = [
                    str(c) if c is not None else f"Column_{i}"
                    for i, c in enumerate(raw_rows[0])
                ]
                rows = [list(r) for r in raw_rows[1:]]
                sheets[sheet_name] = {"headers": headers, "rows": rows}
        finally:
            wb.close()

        return sheets

    @staticmethod
    def parse_sheets(file_record: FileRecord) -> dict[str, DataTable]:
        """Parse all sheets and return a dict of sheet_name -> DataTable."""
        sheets = ExcelParser.read_sheets(file_record)
        tables: dict[str, DataTable] = {}
        for sheet_name, data in sheets.items():
            headers = data["headers"]
            rows = data["rows"]
            table = DataTable(
                name=sheet_name,
                source=f"{file_record.original_name}:{sheet_name}",
                row_count=len(rows),
                column_count=len(headers),
            )
            for idx, header in enumerate(headers):
                col = CsvParser._profile_column(
                    idx, header, [CsvParser._cell_to_str(r, idx) for r in rows]
                )
                table.columns.append(col.column_id)
                table.metadata.setdefault("columns", []).append(col.model_dump())
            tables[sheet_name] = table
        return tables

    @staticmethod
    def parse(file_record: FileRecord) -> tuple[list[DataTable], list[DataProfile]]:
        """Parse an Excel file and return tables + real profiles."""
        tables = []
        profiles = []

        for table in ExcelParser.parse_sheets(file_record).values():
            tables.append(table)
            column_profiles = [
                ColumnProfile.model_validate(c)
                for c in table.metadata.get("columns", [])
            ]
            profiles.append(DataProfile(
                dataset_id="",
                table_id=table.table_id,
                row_count=table.row_count,
                column_count=table.column_count,
                columns=column_profiles,
                total_missing=sum(c.missing_count for c in column_profiles),
            ))

        return tables, profiles


# ═══════════════════════════════════════════════════════════════
# PDF Parser
# ═══════════════════════════════════════════════════════════════

class PdfParser:
    """Parser for PDF files. Extracts text and metadata."""

    @staticmethod
    def parse(file_record: FileRecord) -> dict[str, Any]:
        """Parse a PDF and return structured content."""
        path = Path(file_record.storage_path)
        reader_cls = None
        reader_name = ""
        for module_name in ("pypdf", "PyPDF2"):
            try:
                module = __import__(module_name)
                reader_cls = module.PdfReader
                reader_name = module_name
                break
            except ImportError:
                continue

        if reader_cls is None:
            try:
                import pdfplumber  # noqa: F401
                return PdfParser._parse_with_pdfplumber(path)
            except ImportError:
                return PdfParser._parse_metadata_only(path)

        try:
            reader = reader_cls(str(path))
            page_count = len(reader.pages)
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append({
                    "page_number": i + 1,
                    "text": text,
                    "char_count": len(text),
                })

            return {
                "page_count": page_count,
                "pages": pages,
                "total_chars": sum(p["char_count"] for p in pages),
                "metadata": dict(reader.metadata or {}),
                "is_image_only": all(p["char_count"] == 0 for p in pages),
                "parser": reader_name,
            }
        except Exception as e:
            return {
                "page_count": 0,
                "pages": [],
                "error": str(e),
                "parser": "failed",
            }

    @staticmethod
    def _parse_with_pdfplumber(path: Path) -> dict[str, Any]:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append({
                    "page_number": i + 1,
                    "text": text,
                    "char_count": len(text),
                })
            return {
                "page_count": len(pages),
                "pages": pages,
                "total_chars": sum(p["char_count"] for p in pages),
                "is_image_only": all(p["char_count"] == 0 for p in pages),
                "parser": "pdfplumber",
            }

    @staticmethod
    def _parse_metadata_only(path: Path) -> dict[str, Any]:
        return {
            "page_count": 0,
            "pages": [],
            "error": "No PDF parser available (install PyPDF2 or pdfplumber)",
            "parser": "none",
            "is_image_only": True,
        }


# ═══════════════════════════════════════════════════════════════
# TXT Parser
# ═══════════════════════════════════════════════════════════════

class TxtParser:
    """Parser for plain text files."""

    @staticmethod
    def parse(file_record: FileRecord) -> dict[str, Any]:
        """Parse a TXT file and return content."""
        path = Path(file_record.storage_path)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")

        lines = content.splitlines()
        return {
            "line_count": len(lines),
            "char_count": len(content),
            "content": content,
            "preview": content[:1000],
        }