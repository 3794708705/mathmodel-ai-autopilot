"""MathModel AI — ProblemState ORM model.

The unified state object that all agents operate on.
This is the Phase 1 skeleton — fields will be expanded in later phases.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Enum, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from mathmodel.database import Base
from mathmodel.models.base import TimestampMixin


class ProblemStateStage(str, enum.Enum):
    """Workflow stages for a problem."""
    INGEST = "ingest"
    UNDERSTAND = "understand"
    DATA = "data"
    LITERATURE = "literature"
    EXPLORE = "explore"
    SELECT = "select"
    MODEL = "model"
    SOLVE = "solve"
    VALIDATE = "validate"
    SENSITIVITY = "sensitivity"
    ROBUSTNESS = "robustness"
    RED_TEAM = "red_team"
    PAPER = "paper"
    FINAL_JURY = "final_jury"
    SUBMISSION = "submission"
    FINAL = "final"


class ProblemStateStatus(str, enum.Enum):
    """Status of a problem state or stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProblemState(Base, TimestampMixin):
    """Unified state for a mathematical modeling problem.

    All agents read from and write to this single state object.
    No agent maintains private facts.
    """

    __tablename__ = "problem_states"

    # ── Primary Key ──────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )

    # ── Identification ───────────────────────────────────────
    project_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    competition: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    deadline: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Raw Input ────────────────────────────────────────────
    raw_problem: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    files: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # ── Problem Understanding ────────────────────────────────
    background: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    objectives: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    subproblems: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )

    # ── Facts, Data, Assumptions ─────────────────────────────
    facts: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    data_sources: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    assumptions: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    ambiguities: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    constraints: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )

    # ── Model Selection ──────────────────────────────────────
    candidate_models: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    selected_model: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    backup_model: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    # ── Mathematical Model ───────────────────────────────────
    variables: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    parameters: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    equations: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )

    # ── Computation ──────────────────────────────────────────
    code_files: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    execution_records: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    results: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # ── Verification ─────────────────────────────────────────
    validation_results: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    sensitivity_results: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    robustness_results: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    # ── Literature ───────────────────────────────────────────
    literature: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    citations: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )

    # ── Review & Revision ────────────────────────────────────
    red_team_reports: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    revisions: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )

    # ── Output ───────────────────────────────────────────────
    figures: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    tables: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    paper_state: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    submission_state: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    # ── Workflow State ───────────────────────────────────────
    current_stage: Mapped[ProblemStateStage] = mapped_column(
        Enum(ProblemStateStage), default=ProblemStateStage.INGEST, nullable=False
    )
    status: Mapped[ProblemStateStatus] = mapped_column(
        Enum(ProblemStateStatus), default=ProblemStateStatus.PENDING, nullable=False
    )
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stage_history: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )

    # ── Metadata ─────────────────────────────────────────────
    remaining_hours: Mapped[Optional[float]] = mapped_column(nullable=True)
    metadata_: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata", JSON, nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<ProblemState(id={self.id}, stage={self.current_stage.value}, "
            f"status={self.status.value})>"
        )