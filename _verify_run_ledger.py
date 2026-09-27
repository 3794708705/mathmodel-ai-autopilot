"""Verify the run ledger against a real archived run and a dead database.

Two things are checked here that the hermetic tests cannot check:

1. A real run's recorded state, together with the artifacts it left on disk,
   becomes a faithful database row. The run used is the one the CUMCM 2026 A
   deliverable was selected from — it ended blocked at verification, so the
   row must say exactly that, with the real statistics and the real verdict.
2. With no database reachable (the project's default PostgreSQL URL), a run
   still completes and the run state records that writeback was disabled.

Run from the project root:  .venv\\Scripts\\python.exe _verify_run_ledger.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from mathmodel.autopilot.ledger import RunLedger, problem_stage_for  # noqa: E402
from mathmodel.autopilot.state import RunState  # noqa: E402
from mathmodel.database import Base  # noqa: E402
from mathmodel.models.problem_state import (  # noqa: E402
    ProblemState,
    ProblemStateStage,
    ProblemStateStatus,
)

REAL_RUN_DIR = PROJECT_ROOT / "runs" / "cumcm-2026-a2026-formulafix3"

_results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    _results.append((label, bool(condition), detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))


def replay_real_run(tmp_dir: Path) -> None:
    print(f"\n1. Replaying the real run at {REAL_RUN_DIR.name}")

    if not (REAL_RUN_DIR / "pipeline_state.json").exists():
        check("the archived run exists", False, str(REAL_RUN_DIR))
        return

    state = RunState.load(REAL_RUN_DIR)
    print(f"     run_id={state.run_id} status={state.status.value} "
          f"current_stage={state.current_stage} stages={len(state.stages)}")

    engine = create_engine(f"sqlite:///{tmp_dir / 'ledger.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)

    RunLedger(state, session_factory=factory)
    state.notify()  # the run is already finished; sync what it recorded

    with factory() as session:
        rows = list(session.execute(select(ProblemState)).scalars())
    engine.dispose()

    check("exactly one row for the run", len(rows) == 1, f"{len(rows)} row(s)")
    if not rows:
        return
    row = rows[0]

    check("row is keyed by the run id", row.run_id == state.run_id, str(row.run_id))
    check(
        "run status maps onto the problem status",
        row.status == ProblemStateStatus.FAILED,
        f"{row.status.value} (run was {state.status.value})",
    )
    check(
        "current stage maps onto the workflow vocabulary",
        row.current_stage == ProblemStateStage.VALIDATE,
        f"{row.current_stage.value} (runtime stage {state.current_stage!r})",
    )

    history = row.stage_history or []
    check(
        "every runtime stage is recorded",
        len(history) == len(state.stages),
        f"{len(history)} of {len(state.stages)}",
    )
    unmapped = [e["stage"] for e in history if e["problem_stage"] is None]
    check("no stage is left without a workflow phase", not unmapped, str(unmapped))
    wrong = [
        e["stage"]
        for e in history
        if e["problem_stage"]
        and problem_stage_for(e["stage"]).value != e["problem_stage"]
    ]
    check("recorded phases match the mapping table", not wrong, str(wrong))
    print("     history: " + ", ".join(
        f"{e['stage']}={e['problem_stage']}({e['status']})" for e in history
    ))

    blocked = [e for e in history if e["status"] == "blocked"]
    check(
        "the blocked verification is recorded as such",
        any(e["stage"] == "verify" for e in blocked),
        f"{len(blocked)} blocked stage(s)",
    )

    results = row.results or {}
    check(
        "results come from the newest solve attempt",
        results.get("source") == "solve_outcome" and results.get("attempt") == 6,
        f"source={results.get('source')} attempt={results.get('attempt')}",
    )
    check(
        "the real statistics are recorded",
        bool(results.get("statistics")),
        f"{len(results.get('statistics') or {})} statistic(s)",
    )
    check(
        "the produced result files are recorded",
        len(results.get("result_files") or []) == 4,
        str(results.get("result_files")),
    )
    if results.get("statistics"):
        sample = list(results["statistics"].items())[:3]
        print(f"     statistics sample: {sample}")

    validation = row.validation_results or {}
    report = validation.get("report") or {}
    check(
        "the verdict that gated the run is recorded",
        validation.get("report_file") == "report6.json" and report.get("overall") == "FAIL",
        f"{validation.get('report_file')} overall={report.get('overall')} "
        f"checks={len(report.get('checks') or [])}",
    )
    check(
        "the last error is readable from the row",
        "verify" in (row.error_message or ""),
        (row.error_message or "")[:80],
    )

    # A replay cannot restore objects that only existed in the live process.
    print(f"     metadata mirrored from the live run: {row.metadata_ is not None} "
          "(a replay has no live ProblemState, so null is expected)")


def dead_database_is_survivable(tmp_dir: Path) -> None:
    from mathmodel.config import get_settings

    url = get_settings().database_url
    print(f"\n2. Running with no reachable database ({url.split('@')[-1]})")

    run_dir = tmp_dir / "run"
    state = RunState.create(run_dir)
    ledger = RunLedger(state)  # no injected factory: uses the configured database

    try:
        state.begin("intake", "reading problem statement and attachments")
        state.complete("intake", "2 file(s)")
        survived = True
    except Exception as exc:  # a database must never reach the run
        survived = False
        print(f"     raised: {type(exc).__name__}: {exc}")

    check("the run kept working through an unusable database", survived)
    check("the ledger disabled itself", ledger.disabled)
    reloaded = RunState.load(run_dir)
    check(
        "the file that resumes the run was still written",
        reloaded.stages["intake"].status.value == "completed",
        reloaded.stages["intake"].status.value,
    )
    note = next((n for n in reloaded.notes if "run ledger" in n), "")
    check("the run state records that writeback was disabled", bool(note), note[:120])


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        replay_real_run(tmp_dir)
        dead_database_is_survivable(tmp_dir)

    failed = [label for label, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("failed checks:")
        for label in failed:
            print(f"  - {label}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
