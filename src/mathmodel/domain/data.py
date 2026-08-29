"""MathModel AI — File domain models.

Typed schemas for file records, datasets, tables, and columns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# FileRecord
# ═══════════════════════════════════════════════════════════════

class ParserStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    COMPLETED = "completed"
    FAILED = "failed"
    OCR_REQUIRED = "ocr_required"
    MULTIMODAL_REQUIRED = "multimodal_required"
    UNSUPPORTED = "unsupported"


class FileRecord(BaseModel):
    """Record of an uploaded/ingested file.

    Original files are immutable — never overwritten.
    """

    file_id: str = Field(default_factory=lambda: f"FILE-{uuid4().hex[:8]}")
    project_id: Optional[str] = None
    problem_id: Optional[str] = None

    # File identity
    original_name: str = Field(..., min_length=1)
    safe_name: str = Field(default="")
    media_type: str = Field(default="application/octet-stream")
    extension: str = Field(default="")
    size_bytes: int = Field(default=0, ge=0)
    sha256: str = Field(default="")

    # Storage
    storage_path: str = Field(default="")
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Parsing
    parser_status: ParserStatus = ParserStatus.PENDING
    parser_name: Optional[str] = None
    parse_errors: list[str] = Field(default_factory=list)

    # Provenance
    is_original: bool = True  # True for uploaded files, False for derived

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class DataLayer(str, Enum):
    RAW = "raw"
    CLEANED = "cleaned"
    PROCESSED = "processed"


class Dataset(BaseModel):
    """A dataset composed of one or more tables.

    Tracks provenance from source files through processing steps.
    """

    dataset_id: str = Field(default_factory=lambda: f"DS-{uuid4().hex[:8]}")
    source_file_ids: list[str] = Field(default_factory=list)
    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    layer: DataLayer = DataLayer.RAW
    tables: list[str] = Field(default_factory=list, description="table_ids")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    parent_dataset_id: Optional[str] = None
    processing_steps: list[str] = Field(default_factory=list)
    data_quality: Optional[dict[str, Any]] = None

    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# DataTable
# ═══════════════════════════════════════════════════════════════

class DataTable(BaseModel):
    """A single table within a dataset."""

    table_id: str = Field(default_factory=lambda: f"TBL-{uuid4().hex[:8]}")
    name: str = Field(..., min_length=1)
    source: str = Field(default="", description="Source file or sheet name")
    row_count: int = 0
    column_count: int = 0
    columns: list[str] = Field(default_factory=list, description="column_ids")

    # Semantic hints
    primary_time_field: Optional[str] = None
    primary_spatial_field: Optional[str] = None
    target_candidates: list[str] = Field(default_factory=list)
    feature_candidates: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Column
# ═══════════════════════════════════════════════════════════════

class SemanticType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DURATION = "duration"
    UNKNOWN = "unknown"


class ColumnProfile(BaseModel):
    """Profile of a single column in a data table."""

    column_id: str = Field(default_factory=lambda: f"COL-{uuid4().hex[:8]}")
    name: str = Field(..., min_length=1)
    original_name: str = Field(default="")
    dtype: str = Field(default="object")
    semantic_type: SemanticType = SemanticType.UNKNOWN
    unit: Optional[str] = None
    nullable: bool = True
    missing_count: int = 0
    missing_rate: float = 0.0
    unique_count: Optional[int] = None

    # Numeric statistics (Optional — only for numeric columns)
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    quantiles: Optional[dict[str, float]] = None

    # Example values
    example_values: list[Any] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# DataProfile
# ═══════════════════════════════════════════════════════════════

class DataProfile(BaseModel):
    """Complete profile of a dataset, computed by Python (not LLM)."""

    profile_id: str = Field(default_factory=lambda: f"PROF-{uuid4().hex[:8]}")
    dataset_id: str
    table_id: str

    # Shape
    row_count: int = 0
    column_count: int = 0

    # Columns
    columns: list[ColumnProfile] = Field(default_factory=list)

    # Quality
    total_missing: int = 0
    duplicate_rows: int = 0
    potential_key_duplicates: int = 0

    # Outlier candidates
    outlier_candidates: list[dict[str, Any]] = Field(default_factory=list)

    # Correlation candidates
    correlation_candidates: list[dict[str, Any]] = Field(default_factory=list)

    # Time
    time_range: Optional[dict[str, Any]] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    computation_time_ms: float = 0.0

    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# DataCleaningPlan
# ═══════════════════════════════════════════════════════════════

class CleaningOperation(BaseModel):
    """A single cleaning operation in a plan."""

    operation_id: str = Field(default_factory=lambda: f"OP-{uuid4().hex[:8]}")
    target: str = Field(..., description="column_id or table_id")
    operation: str = Field(..., description="drop, fill_mean, fill_median, etc.")
    reason: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    risk: str = Field(default="low", description="low, medium, high")
    reversible: bool = True


class DataCleaningPlan(BaseModel):
    """A plan for cleaning data, proposed by DataAgent."""

    plan_id: str = Field(default_factory=lambda: f"DCP-{uuid4().hex[:8]}")
    dataset_id: str
    operations: list[CleaningOperation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)