"""MathModel AI — CUMCM Autopilot run state.

One continuous workflow with durable state, so a run can be paused at a
real decision point and resumed without repeating completed work.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr


def _now() -> datetime:
    return datetime.now(timezone.utc)


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"              # waiting for a clarification answer
    NEEDS_CONFIRMATION = "needs_confirmation"  # waiting for a modeling decision
    BLOCKED = "blocked"
    COMPLETED = "completed"


class ClarificationQuestion(BaseModel):
    """A question asked only because information is genuinely missing."""

    question_id: str
    question: str
    reason: str = ""
    blocking: bool = True
    options: list[str] = Field(default_factory=list)


class StageRecord(BaseModel):
    name: str
    status: StageStatus = StageStatus.PENDING
    detail: str = ""
    attempts: int = 0
    artifacts: list[str] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: str = ""


class RunState(BaseModel):
    """Durable state of one Autopilot run."""

    run_id: str = Field(
        default_factory=lambda: (
            f"cumcm-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
        )
    )
    run_dir: str = ""
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    status: RunStatus = RunStatus.RUNNING
    current_stage: str = ""
    stages: dict[str, StageRecord] = Field(default_factory=dict)

    input_files: list[str] = Field(default_factory=list)
    pending_questions: list[ClarificationQuestion] = Field(default_factory=list)
    answers: dict[str, str] = Field(default_factory=dict)

    counters: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    # Process-local wiring, deliberately kept out of serialization: a loaded
    # state must not carry a callback that belonged to another process.
    _watcher: Optional[Callable[["RunState"], None]] = PrivateAttr(default=None)

    # ── Change notification ───────────────────────────────────

    def watch(self, callback: Optional[Callable[["RunState"], None]]) -> None:
        """Register a callback invoked after every change to this run."""
        self._watcher = callback

    def notify(self) -> None:
        """Announce the current state to the watcher, if one is registered."""
        if self._watcher is not None:
            self._watcher(self)

    # ── Stage helpers ─────────────────────────────────────────

    def stage(self, name: str) -> StageRecord:
        record = self.stages.get(name)
        if record is None:
            record = StageRecord(name=name)
            self.stages[name] = record
        return record

    def begin(self, name: str, detail: str = "") -> StageRecord:
        record = self.stage(name)
        record.status = StageStatus.RUNNING
        record.attempts += 1
        record.detail = detail
        record.started_at = _now()
        record.error = ""
        self.current_stage = name
        self.updated_at = _now()
        self.save()
        return record

    def complete(
        self, name: str, detail: str = "", artifacts: Optional[list[str]] = None
    ) -> StageRecord:
        record = self.stage(name)
        record.status = StageStatus.COMPLETED
        record.detail = detail
        record.finished_at = _now()
        if artifacts:
            record.artifacts = list(artifacts)
        self.updated_at = _now()
        self.save()
        return record

    def fail(self, name: str, error: str) -> StageRecord:
        record = self.stage(name)
        record.status = StageStatus.FAILED
        record.error = error
        record.finished_at = _now()
        self.updated_at = _now()
        self.save()
        return record

    def block(self, name: str, reason: str) -> StageRecord:
        record = self.stage(name)
        record.status = StageStatus.BLOCKED
        record.detail = reason
        record.finished_at = _now()
        self.status = RunStatus.BLOCKED
        self.updated_at = _now()
        self.save()
        return record

    def skip(self, name: str, reason: str) -> StageRecord:
        record = self.stage(name)
        record.status = StageStatus.SKIPPED
        record.detail = reason
        record.finished_at = _now()
        self.updated_at = _now()
        self.save()
        return record

    def is_done(self, name: str) -> bool:
        return self.stage(name).status == StageStatus.COMPLETED

    def count(self, key: str, delta: int = 1) -> int:
        self.counters[key] = self.counters.get(key, 0) + delta
        return self.counters[key]

    # ── Persistence ───────────────────────────────────────────

    @property
    def state_path(self) -> Path:
        return Path(self.run_dir) / "pipeline_state.json"

    def save(self) -> None:
        self.updated_at = _now()
        path = self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(path)
        self.notify()

    @classmethod
    def load(cls, run_dir: str | Path) -> "RunState":
        path = Path(run_dir) / "pipeline_state.json"
        if not path.exists():
            raise FileNotFoundError(f"No pipeline state at {path}")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def create(cls, run_dir: str | Path) -> "RunState":
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        state = cls(run_dir=str(run_dir))
        state.save()
        return state

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "current_stage": self.current_stage,
            "stages": {
                name: {"status": rec.status.value, "detail": rec.detail}
                for name, rec in self.stages.items()
            },
            "pending_questions": [q.model_dump() for q in self.pending_questions],
            "counters": self.counters,
            "notes": self.notes,
        }
