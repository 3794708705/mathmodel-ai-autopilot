"""MathModel AI — Phase 6P: Code generation domain schemas.

CodeArtifact, CodeMapping, CodeGenerationResult.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class CodeObjectType(str, Enum):
    EQUATION = "equation"
    OBJECTIVE = "objective"
    CONSTRAINT = "constraint"
    VARIABLE = "variable"
    PARAMETER = "parameter"


class GeneratedFile(BaseModel):
    """A single generated code file."""
    path: str = Field(..., min_length=1)
    purpose: str = Field(default="")
    hash: str = Field(default="")
    equations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)


class CodeArtifact(BaseModel):
    """A complete generated code artifact."""
    code_artifact_id: str = Field(default_factory=lambda: f"CODE-{uuid4().hex[:8]}")
    model_id: str = ""
    version: int = 1
    files: list[GeneratedFile] = Field(default_factory=list)
    language: str = "python"
    entrypoint: str = ""
    dependencies: list[str] = Field(default_factory=list)
    code_hash: str = ""
    generated_by: str = "CodeAgent"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_status: str = "not_executed"


class CodeMapping(BaseModel):
    """Maps a mathematical object to generated code."""
    mapping_id: str = Field(default_factory=lambda: f"CM-{uuid4().hex[:8]}")
    model_id: str = ""
    mathematical_object_id: str = Field(..., min_length=1)
    object_type: CodeObjectType
    code_artifact_id: str = ""
    file: str = ""
    function: str = ""
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    generated_representation: str = ""
    code_hash: str = ""


class CodeGenerationResult(BaseModel):
    """Result of CodeAgent run."""
    artifact: Optional[CodeArtifact] = None
    mappings: list[CodeMapping] = Field(default_factory=list)
    status: str = "completed"  # completed | blocked
    blocked_reason: str = ""
    warnings: list[str] = Field(default_factory=list)