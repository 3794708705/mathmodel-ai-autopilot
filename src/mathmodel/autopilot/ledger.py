"""MathModel AI — run ledger: database writeback for autopilot runs.

`<run_dir>/pipeline_state.json` remains the mechanism that resumes a run.
This ledger makes the database a faithful record of the *same* run — every
stage transition, the run status, the verification verdict and the produced
results — so `problem_states` is populated by real runs instead of only by
the REST CRUD endpoints.

Two properties are load-bearing:

* **Non-fatal.** An unreachable database, a missing table or a missing
  driver must never abort a run that would otherwise succeed. The first
  failure disables the ledger for the rest of the process, logs it, and
  leaves a note in the run state, so the record says the database was not
  written rather than pretending it was.
* **Derived, never duplicated.** Stage data comes from the `RunState` the
  pipeline already maintains; `results` and `validation_results` are read
  from the artifacts the run already wrote. The ledger keeps no facts of its
  own, so it cannot disagree with the run.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy import select

from mathmodel.autopilot.state import RunState, RunStatus, StageStatus
from mathmodel.models.problem_state import (
    ProblemState,
    ProblemStateStage,
    ProblemStateStatus,
)

logger = logging.getLogger(__name__)


# ── Vocabulary mapping ───────────────────────────────────────
#
# The runtime names its stages after what it is doing (`codegen`, `audit`);
# the workflow vocabulary in `ProblemStateStage` names the modelling phase.
# The two are different languages for the same run, so every runtime stage is
# mapped explicitly — an unmapped stage is recorded in `stage_history` under
# its own name but must never be given a guessed phase.

RUNTIME_STAGE_TO_PROBLEM_STAGE: dict[str, ProblemStateStage] = {
    "intake": ProblemStateStage.INGEST,
    "clarify": ProblemStateStage.INGEST,
    "understand": ProblemStateStage.UNDERSTAND,
    "registries": ProblemStateStage.UNDERSTAND,
    "explore": ProblemStateStage.EXPLORE,
    "select": ProblemStateStage.SELECT,
    "model": ProblemStateStage.MODEL,
    "codegen": ProblemStateStage.SOLVE,
    "solve": ProblemStateStage.SOLVE,
    "verify": ProblemStateStage.VALIDATE,
    "evidence": ProblemStateStage.VALIDATE,
    "figures": ProblemStateStage.PAPER,
    "tables": ProblemStateStage.PAPER,
    "paper": ProblemStateStage.PAPER,
    "audit": ProblemStateStage.FINAL_JURY,
    "pdf": ProblemStateStage.SUBMISSION,
    "package": ProblemStateStage.SUBMISSION,
    "final_check": ProblemStateStage.FINAL,
}

RUN_STATUS_TO_PROBLEM_STATUS: dict[RunStatus, ProblemStateStatus] = {
    RunStatus.RUNNING: ProblemStateStatus.RUNNING,
    RunStatus.NEEDS_INPUT: ProblemStateStatus.PENDING,
    RunStatus.NEEDS_CONFIRMATION: ProblemStateStatus.PENDING,
    RunStatus.BLOCKED: ProblemStateStatus.FAILED,
    RunStatus.COMPLETED: ProblemStateStatus.COMPLETED,
}

# Stages beyond this are recorded but not mapped, which keeps a new runtime
# stage visible in the history instead of silently mislabelled.
_MAX_TEXT = 2000
_MAX_TITLE = 500

_SCHEMA_CHECKED = False


def problem_stage_for(runtime_stage: str) -> Optional[ProblemStateStage]:
    """Map a runtime stage name onto the workflow stage vocabulary."""
    return RUNTIME_STAGE_TO_PROBLEM_STAGE.get(runtime_stage)


def problem_status_for(status: RunStatus) -> ProblemStateStatus:
    """Map a run status onto the problem state status vocabulary."""
    return RUN_STATUS_TO_PROBLEM_STATUS.get(status, ProblemStateStatus.RUNNING)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _json_default(value: Any) -> str:
    """How a value the JSON encoder cannot handle is recorded."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _json_safe(value: Any) -> Any:
    """A value the row's JSON columns can actually store.

    Everything the ledger writes to `problem_states` passes through here. The
    domain object is free to carry datetimes, UUIDs or enums in its metadata,
    and a single such value used to abort the whole write with "Object of type
    datetime is not JSON serializable", which disabled the ledger for the rest
    of the run.
    """
    return json.loads(json.dumps(value, default=_json_default, ensure_ascii=False))


def _attempt_number(filename: str) -> Optional[int]:
    match = re.search(r"outcome(\d+)\.json$", filename)
    return int(match.group(1)) if match else None


def _last_error(run_state: RunState) -> str:
    """Why the run stopped, from the last stage that failed or blocked.

    A blocked stage is as much a reason to stop as a failed one: a run that
    exhausts its repair attempts blocks verification rather than failing it,
    and the row must still say why it stopped.
    """
    for record in reversed(list(run_state.stages.values())):
        if record.status == StageStatus.FAILED:
            reason = record.error or record.detail
        elif record.status == StageStatus.BLOCKED:
            # `block()` carries the run-level reason in `detail`; the stage may
            # still hold the per-attempt error from an earlier failure, which
            # stays visible in the stage history.
            reason = record.detail or record.error
        else:
            continue
        if reason:
            return f"{record.name}: {reason}"[:_MAX_TEXT]
    return ""


class RunLedger:
    """Keeps one autopilot run's `ProblemState` row in sync with the run."""

    def __init__(
        self,
        run_state: RunState,
        session_factory: Optional[Callable[[], Any]] = None,
    ):
        self._run_id = run_state.run_id
        self._run_dir = Path(run_state.run_dir)
        self._session_factory = session_factory
        self._domain_state: Optional[ProblemState] = None
        self._disabled = False
        self._last_signature: Optional[str] = None
        self.writes = 0
        run_state.watch(self.sync)

    # ── Wiring ────────────────────────────────────────────────

    def observe_domain_state(self, state_obj: ProblemState) -> None:
        """Register the in-memory `ProblemState` whose reasoning to mirror."""
        self._domain_state = state_obj

    @property
    def disabled(self) -> bool:
        return self._disabled

    # ── Sync ──────────────────────────────────────────────────

    def sync(self, run_state: RunState) -> None:
        """Mirror the run into the database. This method never raises."""
        if self._disabled:
            return
        try:
            payload = self._build_payload(run_state)
            signature = json.dumps(
                payload, sort_keys=True, ensure_ascii=False, default=str
            )
            if signature == self._last_signature:
                # Nothing about the run changed, so the row is already correct.
                return
            self._write(payload)
            self._last_signature = signature
            self.writes += 1
        except Exception as exc:  # a run must survive an unusable database
            self._disable(run_state, exc)

    # ── Payload ───────────────────────────────────────────────

    def _build_payload(self, run_state: RunState) -> dict[str, Any]:
        domain = self._domain_state
        stages: list[dict[str, Any]] = []
        for name, record in run_state.stages.items():
            stage = problem_stage_for(name)
            if stage is None:
                logger.debug(
                    "Run %s stage %r has no workflow phase mapping", self._run_id, name
                )
            stages.append(
                {
                    "stage": name,
                    "problem_stage": stage.value if stage else None,
                    "status": record.status.value,
                    "detail": record.detail[:_MAX_TEXT],
                    "error": record.error[:_MAX_TEXT],
                    "attempts": record.attempts,
                    "started_at": _iso(record.started_at),
                    "finished_at": _iso(record.finished_at),
                    "artifacts": list(record.artifacts),
                }
            )

        current_stage = problem_stage_for(run_state.current_stage)

        # The artifacts are the run's own output and win over the in-memory
        # object; the object is the fallback so a value written there by an
        # agent is never blanked out by a run that produced no manifest.
        results = self._results_payload(run_state)
        if results is None and domain is not None:
            results = getattr(domain, "results", None)
        validation = self._validation_payload()
        if validation is None and domain is not None:
            validation = getattr(domain, "validation_results", None)

        return {
            "run_id": run_state.run_id,
            "run_dir": run_state.run_dir,
            "run_status": run_state.status.value,
            "problem_status": problem_status_for(run_state.status).value,
            "current_stage": run_state.current_stage,
            "problem_stage": current_stage.value if current_stage else None,
            "stages": stages,
            "notes": list(run_state.notes),
            "error_message": _last_error(run_state),
            "title": getattr(domain, "title", None),
            "competition": getattr(domain, "competition", None),
            "raw_problem": getattr(domain, "raw_problem", None),
            "metadata": dict(getattr(domain, "metadata_", None) or {}),
            "results": results,
            "validation_results": validation,
        }

    def _results_payload(self, run_state: RunState) -> Optional[dict[str, Any]]:
        """Read the run's results from what the run already wrote.

        A packaged run is described by its support-package manifest. A run that
        stopped before packaging — most runs while a model is being fixed —
        still produced results, so its newest solve outcome is recorded
        instead. The solver's `per_subproblem` tables are left out on purpose:
        they are solver detail, not the run's results.
        """
        manifest = self._read_json(self._run_dir / "output" / "manifest.json")
        if isinstance(manifest, dict):
            return {
                "source": "support_package",
                "run_id": manifest.get("run_id") or run_state.run_id,
                "run_status": run_state.status.value,
                "generated_at": manifest.get("generated_at"),
                "paper_pdf": manifest.get("paper_pdf"),
                "paper_pdf_size": manifest.get("paper_pdf_size"),
                "competition": manifest.get("competition"),
                "model": manifest.get("model"),
                "statistics": manifest.get("statistics"),
                "result_files": manifest.get("result_files"),
                "figures": manifest.get("figures"),
                "tables": manifest.get("tables"),
                "counts": manifest.get("counts"),
                "submission_check": manifest.get("submission_check"),
                "data_status": manifest.get("data_status"),
            }

        outcome, name = self._newest_solve_outcome()
        if outcome is None:
            return None
        summary = outcome.get("summary") or {}
        return {
            "source": "solve_outcome",
            "run_id": run_state.run_id,
            "run_status": run_state.status.value,
            "attempt": _attempt_number(name),
            "solve_status": outcome.get("status"),
            "exit_code": outcome.get("exit_code"),
            "runtime_seconds": outcome.get("runtime_seconds"),
            "execution_real": outcome.get("execution_real"),
            "output_dir": outcome.get("output_dir"),
            "result_files": outcome.get("artifacts") or [],
            "statistics": summary.get("statistics") or {},
            "self_check": summary.get("self_check") or {},
        }

    def _newest_solve_outcome(self) -> tuple[Optional[dict[str, Any]], str]:
        directory = self._run_dir / "artifacts" / "solve"
        outcomes = [
            path for path in directory.glob("outcome*.json") if path.is_file()
        ]
        if not outcomes:
            return None, ""
        latest = max(outcomes, key=lambda path: path.stat().st_mtime)
        outcome = self._read_json(latest)
        if not isinstance(outcome, dict):
            return None, ""
        return outcome, latest.name

    def _validation_payload(self) -> Optional[dict[str, Any]]:
        """Read the verdict that gated the run from the newest verify report.

        Reports are numbered per attempt; the newest file is the attempt that
        decided this run, matching what `CUMCMAutopilot._result` reports.
        """
        directory = self._run_dir / "artifacts" / "verify"
        reports = [path for path in directory.glob("report*.json") if path.is_file()]
        if not reports:
            return None
        latest = max(reports, key=lambda path: path.stat().st_mtime)
        report = self._read_json(latest)
        if report is None:
            return None
        return {"report_file": latest.name, "report": report}

    @staticmethod
    def _read_json(path: Path) -> Optional[Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # ── Database ──────────────────────────────────────────────

    def _write(self, payload: dict[str, Any]) -> None:
        session_factory = self._session_factory_for()
        with session_factory() as session:
            row = session.execute(
                select(ProblemState).where(ProblemState.run_id == self._run_id)
            ).scalar_one_or_none()
            if row is None:
                row = ProblemState(run_id=self._run_id)
                session.add(row)

            if payload.get("title"):
                row.title = str(payload["title"])[:_MAX_TITLE]
            if payload.get("competition"):
                row.competition = str(payload["competition"])[:_MAX_TITLE]
            if payload.get("raw_problem"):
                row.raw_problem = str(payload["raw_problem"])

            if payload.get("problem_stage"):
                row.current_stage = ProblemStateStage(payload["problem_stage"])
            row.status = ProblemStateStatus(payload["problem_status"])
            row.stage_history = _json_safe(payload["stages"])
            row.metadata_ = _json_safe(payload["metadata"]) or None
            row.results = _json_safe(payload["results"])
            row.validation_results = _json_safe(payload["validation_results"])
            row.error_message = payload["error_message"] or None

            session.commit()

    def _session_factory_for(self) -> Callable[[], Any]:
        if self._session_factory is not None:
            return self._session_factory

        from mathmodel.config import get_settings
        from mathmodel.database import (
            Base,
            get_sync_engine,
            get_sync_session_factory,
        )

        global _SCHEMA_CHECKED
        if not _SCHEMA_CHECKED:
            if get_settings().database_auto_create:
                Base.metadata.create_all(get_sync_engine(), checkfirst=True)
            _SCHEMA_CHECKED = True

        self._session_factory = get_sync_session_factory()
        return self._session_factory

    # ── Failure handling ──────────────────────────────────────

    def _disable(self, run_state: RunState, exc: BaseException) -> None:
        self._disabled = True
        reason = f"{type(exc).__name__}: {exc}"[:_MAX_TEXT]
        logger.warning(
            "Run ledger disabled for run %s: %s", self._run_id, reason
        )
        note = f"[run ledger] database writeback disabled — {reason}"
        if note not in run_state.notes:
            run_state.notes.append(note)
