"""MathModel AI — Phase 6B: Document Registry, Figure and Table systems.

Stable internal IDs (FIG-001, TAB-001, EQ-001, REF-001, CLAIM-001, SEC-001).
Final numbering (图1, Table 2) is the renderer's job, never hand-maintained.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# Document Registry
# ═══════════════════════════════════════════════════════════════

class DocKind(str, Enum):
    SECTION = "section"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    REFERENCE = "reference"
    CLAIM = "claim"
    APPENDIX = "appendix"


class DocumentRegistry:
    """Central registry mapping stable IDs to document objects."""

    def __init__(self):
        self._objects: dict[str, Any] = {}
        self._counters: dict[str, int] = {}

    def register(self, kind: str, obj: Any, explicit_id: Optional[str] = None) -> str:
        """Register an object and return its stable ID (e.g. FIG-001)."""
        if explicit_id:
            if explicit_id in self._objects:
                raise ValueError(f"Duplicate document ID: {explicit_id}")
            doc_id = explicit_id
        else:
            self._counters[kind] = self._counters.get(kind, 0) + 1
            doc_id = f"{kind}-{self._counters[kind]:03d}"
        self._objects[doc_id] = obj
        return doc_id

    def get(self, doc_id: str) -> Optional[Any]:
        return self._objects.get(doc_id)

    def has(self, doc_id: str) -> bool:
        return doc_id in self._objects

    def all_ids(self) -> list[str]:
        return list(self._objects.keys())

    def by_kind(self, kind: str) -> list[str]:
        return sorted(
            [k for k in self._objects if k.startswith(f"{kind}-")],
            key=lambda k: int(k.split("-")[1]) if k.split("-")[1].isdigit() else 0,
        )


# ═══════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════

class FigureType(str, Enum):
    TREND = "trend"
    SCATTER = "scatter"
    BOXPLOT = "boxplot"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    CORRELATION = "correlation"
    COMPARISON = "comparison"
    PREDICTION_VS_ACTUAL = "prediction_vs_actual"
    RESIDUAL = "residual"
    SENSITIVITY = "sensitivity"
    ROBUSTNESS_DISTRIBUTION = "robustness_distribution"
    NETWORK = "network"
    FLOW = "flow"
    SOLUTION = "solution"


class FigureRecord(BaseModel):
    """A paper figure generated from real data or execution."""
    figure_id: str = Field(default_factory=lambda: f"FIG-{uuid4().hex[:8]}")
    title: str = Field(..., min_length=1)
    caption: str = Field(default="")
    figure_type: FigureType = FigureType.TREND
    source_data_ids: list[str] = Field(default_factory=list)
    source_execution_ids: list[str] = Field(default_factory=list)
    generation_code: str = Field(default="")
    artifact_path: str = Field(default="")
    hash: str = Field(default="")
    width: Optional[int] = None
    height: Optional[int] = None
    verification_status: str = "unverified"  # verified | unverified | failed
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class FigureRegistry:
    def __init__(self):
        self._figures: dict[str, FigureRecord] = {}

    def register(self, figure: FigureRecord) -> None:
        if figure.figure_id in self._figures:
            raise ValueError(f"Duplicate figure ID: {figure.figure_id}")
        self._figures[figure.figure_id] = figure

    def get(self, figure_id: str) -> Optional[FigureRecord]:
        return self._figures.get(figure_id)

    def all(self) -> list[FigureRecord]:
        return list(self._figures.values())


def verify_figure(figure: FigureRecord) -> list[str]:
    """Deterministic figure integrity checks."""
    import os
    issues = []
    if not figure.artifact_path:
        issues.append(f"{figure.figure_id}: missing artifact_path")
    elif not os.path.exists(figure.artifact_path):
        issues.append(f"{figure.figure_id}: artifact file does not exist")
    elif os.path.getsize(figure.artifact_path) == 0:
        issues.append(f"{figure.figure_id}: artifact file is empty")
    elif figure.hash:
        # Verify recorded hash matches current file bytes
        import hashlib
        with open(figure.artifact_path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()[:16]
        if actual != figure.hash:
            issues.append(
                f"{figure.figure_id}: hash mismatch (recorded {figure.hash}, actual {actual})"
            )
    if not figure.source_data_ids and not figure.source_execution_ids:
        issues.append(f"{figure.figure_id}: no data or execution source")
    return issues


# ═══════════════════════════════════════════════════════════════
# Table
# ═══════════════════════════════════════════════════════════════

class TableType(str, Enum):
    SYMBOLS = "symbols"
    PARAMETERS = "parameters"
    DESCRIPTIVE_STATS = "descriptive_statistics"
    CANDIDATE_COMPARISON = "candidate_comparison"
    OPTIMIZATION_RESULT = "optimization_result"
    SENSITIVITY = "sensitivity"
    ROBUSTNESS = "robustness"
    VALIDATION = "validation"


class TableCell(BaseModel):
    """A table cell with optional value provenance."""
    value: Any
    source_id: Optional[str] = None  # Evidence/source reference
    source_description: str = ""


class TableRecord(BaseModel):
    """A paper table with source-tracked cells."""
    table_id: str = Field(default_factory=lambda: f"TAB-{uuid4().hex[:8]}")
    title: str = Field(..., min_length=1)
    table_type: TableType = TableType.DESCRIPTIVE_STATS
    headers: list[str] = Field(default_factory=list)
    rows: list[list[TableCell]] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    verification_status: str = "unverified"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableRegistry:
    def __init__(self):
        self._tables: dict[str, TableRecord] = {}

    def register(self, table: TableRecord) -> None:
        if table.table_id in self._tables:
            raise ValueError(f"Duplicate table ID: {table.table_id}")
        self._tables[table.table_id] = table

    def get(self, table_id: str) -> Optional[TableRecord]:
        return self._tables.get(table_id)

    def all(self) -> list[TableRecord]:
        return list(self._tables.values())


def verify_table(
    table: TableRecord,
    source_values: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Deterministic table integrity checks.

    source_values: {source_id: authoritative value}. When provided,
    numeric cells whose source_id is in this map must match the
    authoritative value (formatting tolerance 1e-9).
    """
    issues = []
    source_values = source_values or {}
    for row in table.rows:
        if len(row) != len(table.headers):
            issues.append(
                f"{table.table_id}: row has {len(row)} cells but {len(table.headers)} headers"
            )
        for cell in row:
            if cell.value is not None and cell.source_id is None:
                # Numeric cells should have sources
                if isinstance(cell.value, (int, float)):
                    issues.append(
                        f"{table.table_id}: numeric cell value={cell.value} has no source_id"
                    )
            elif cell.source_id in source_values:
                authoritative = source_values[cell.source_id]
                if isinstance(cell.value, (int, float)) and isinstance(authoritative, (int, float)):
                    if abs(float(cell.value) - float(authoritative)) > 1e-9:
                        issues.append(
                            f"{table.table_id}: cell value {cell.value} does not match "
                            f"source {cell.source_id}={authoritative}"
                        )
    return issues