"""Tests for the autopilot run ledger — the database writeback of a run.

The ledger exists so that a real run, not only the REST CRUD endpoints,
populates `problem_states`. These tests cover the mapping between the
runtime's stage names and the workflow vocabulary, the legality of the
transitions the runtime performs, what the row ends up holding, and the
requirement that an unusable database never breaks a run.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import mathmodel.autopilot.pipeline as pipeline_module
from mathmodel.api.routes import VALID_STAGE_TRANSITIONS
from mathmodel.autopilot.ledger import (
    RUNTIME_STAGE_TO_PROBLEM_STAGE,
    RunLedger,
    problem_stage_for,
    problem_status_for,
)
from mathmodel.autopilot.state import RunState, RunStatus
from mathmodel.database import Base
from mathmodel.models.problem_state import (
    ProblemState,
    ProblemStateStage,
    ProblemStateStatus,
)

# The stages a complete run passes through, in order, as the runtime names
# them. `problem_stage_for` must map every one of them.
CANONICAL_RUN_SEQUENCE = [
    "intake",
    "clarify",
    "understand",
    "registries",
    "explore",
    "select",
    "model",
    "codegen",
    "solve",
    "verify",
    "evidence",
    "figures",
    "tables",
    "paper",
    "audit",
    "pdf",
    "package",
    "final_check",
]


@pytest.fixture
def session_factory():
    """An in-memory SQLite database with the ORM schema created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()


@pytest.fixture
def run_state(tmp_path) -> RunState:
    return RunState.create(tmp_path / "run")


def _rows(session_factory) -> list[ProblemState]:
    with session_factory() as session:
        return list(session.execute(select(ProblemState)).scalars())


# ── Vocabulary ───────────────────────────────────────────────


def test_every_runtime_stage_has_a_workflow_phase():
    """A new runtime stage must not appear without a phase mapping.

    The scan is over the pipeline's own source so that adding a stage is a
    test failure rather than an unmapped `None` in the database.
    """
    source = Path(pipeline_module.__file__).read_text(encoding="utf-8")
    named = set(
        re.findall(
            r'state\.(?:begin|complete|fail|block|skip)\(\s*"([a-z_]+)"', source
        )
    )
    assert named, "no runtime stages found — the scan pattern is stale"

    unmapped = sorted(
        name for name in named if name not in RUNTIME_STAGE_TO_PROBLEM_STAGE
    )
    assert unmapped == [], f"runtime stages without a workflow phase: {unmapped}"


def test_canonical_run_sequence_only_makes_legal_transitions():
    """Every consecutive phase change a complete run performs is allowed."""
    phases: list[ProblemStateStage] = []
    for name in CANONICAL_RUN_SEQUENCE:
        phase = problem_stage_for(name)
        assert phase is not None, f"{name} has no workflow phase"
        if not phases or phases[-1] != phase:
            phases.append(phase)

    assert phases[0] == ProblemStateStage.INGEST
    assert phases[-1] == ProblemStateStage.FINAL
    for current, target in zip(phases, phases[1:]):
        assert target in VALID_STAGE_TRANSITIONS[current], (
            f"{current.value} -> {target.value} is not an allowed transition"
        )


def test_repair_after_verification_failure_is_an_allowed_transition():
    """A failed verification sends the model back for repair."""
    allowed = VALID_STAGE_TRANSITIONS[ProblemStateStage.VALIDATE]
    assert ProblemStateStage.MODEL in allowed


def test_run_status_maps_onto_problem_status():
    assert problem_status_for(RunStatus.RUNNING) == ProblemStateStatus.RUNNING
    assert problem_status_for(RunStatus.COMPLETED) == ProblemStateStatus.COMPLETED
    assert problem_status_for(RunStatus.BLOCKED) == ProblemStateStatus.FAILED
    assert problem_status_for(RunStatus.NEEDS_INPUT) == ProblemStateStatus.PENDING


# ── Row contents ─────────────────────────────────────────────


def test_ledger_creates_one_row_and_records_stage_history(
    run_state, session_factory
):
    RunLedger(run_state, session_factory=session_factory)

    run_state.begin("intake", "reading problem statement and attachments")
    run_state.complete("intake", "2 file(s)")
    run_state.begin("understand", "ProblemAgent (staged)")
    run_state.complete("understand", "3 subproblem(s)", ["analysis.json"])

    rows = _rows(session_factory)
    assert len(rows) == 1
    row = rows[0]
    assert row.run_id == run_state.run_id
    assert row.current_stage == ProblemStateStage.UNDERSTAND
    assert row.status == ProblemStateStatus.RUNNING

    history = {entry["stage"]: entry for entry in row.stage_history}
    assert history["intake"]["status"] == "completed"
    assert history["intake"]["problem_stage"] == "ingest"
    assert history["understand"]["problem_stage"] == "understand"
    assert history["understand"]["artifacts"] == ["analysis.json"]
    assert history["understand"]["finished_at"] is not None


def test_ledger_records_a_failed_stage(run_state, session_factory):
    RunLedger(run_state, session_factory=session_factory)

    run_state.begin("solve", "attempt 1: docker sandbox")
    run_state.fail("solve", "exit code 1")

    row = _rows(session_factory)[0]
    assert row.current_stage == ProblemStateStage.SOLVE
    history = {entry["stage"]: entry for entry in row.stage_history}
    assert history["solve"]["status"] == "failed"
    assert row.error_message == "solve: exit code 1"


def test_ledger_records_a_blocked_stage_reason(run_state, session_factory):
    """A run that blocks at verification must still say why it stopped."""
    RunLedger(run_state, session_factory=session_factory)

    run_state.begin("verify", "attempt 6")
    run_state.fail("verify", "attempt 6 FAILED: 3 issue(s)")
    run_state.block("verify", "verification still failing after 6 repair attempt(s)")

    row = _rows(session_factory)[0]
    assert row.status == ProblemStateStatus.FAILED
    assert row.current_stage == ProblemStateStage.VALIDATE
    # The run-level reason is what the row reports; the per-attempt error is
    # still in the history.
    assert row.error_message == (
        "verify: verification still failing after 6 repair attempt(s)"
    )
    history = {entry["stage"]: entry for entry in row.stage_history}
    assert history["verify"]["error"] == "attempt 6 FAILED: 3 issue(s)"
    assert history["verify"]["status"] == "blocked"


def test_ledger_records_results_and_verification(run_state, session_factory):
    RunLedger(run_state, session_factory=session_factory)
    run_dir = Path(run_state.run_dir)

    (run_dir / "output").mkdir()
    (run_dir / "output" / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_state.run_id,
                "generated_at": "2026-09-27T00:00:00+00:00",
                "paper_pdf": "paper.pdf",
                "paper_pdf_size": 1234,
                "competition": "CUMCM",
                "model": {"model_id": "m1", "name": "heat", "version": 2},
                "statistics": {"peak_remaining": 2.55},
                "result_files": ["result1.xlsx"],
                "figures": [{"figure_id": "fig1", "title": "profile"}],
                "tables": [{"table_id": "t1", "title": "params", "rows": 3}],
                "counts": {"figures": 1},
                "submission_check": {"status": "ready", "failures": []},
                "data_status": "formal",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    verify_dir = run_dir / "artifacts" / "verify"
    verify_dir.mkdir(parents=True)
    (verify_dir / "report1.json").write_text(
        json.dumps({"overall": "FAIL", "checks": []}), encoding="utf-8"
    )
    (verify_dir / "report2.json").write_text(
        json.dumps(
            {
                "overall": "PASS",
                "checks": [{"name": "required_outputs_present", "status": "pass"}],
            }
        ),
        encoding="utf-8",
    )

    # Distinct modification times, so "newest report" cannot depend on how
    # fast the filesystem stamps two writes in the same test.
    os.utime(verify_dir / "report1.json", (1_600_000_000, 1_600_000_000))
    os.utime(verify_dir / "report2.json", (1_700_000_000, 1_700_000_000))

    run_state.begin("final_check", "verifying deliverable consistency")
    run_state.complete("final_check", "all consistency checks passed")
    run_state.status = RunStatus.COMPLETED
    run_state.save()

    row = _rows(session_factory)[0]
    assert row.status == ProblemStateStatus.COMPLETED
    assert row.current_stage == ProblemStateStage.FINAL
    assert row.results["statistics"] == {"peak_remaining": 2.55}
    assert row.results["result_files"] == ["result1.xlsx"]
    assert row.results["model"]["name"] == "heat"
    # The newest report is the attempt that decided the run.
    assert row.validation_results["report_file"] == "report2.json"
    assert row.validation_results["report"]["overall"] == "PASS"


def test_results_fall_back_to_the_newest_solve_outcome(run_state, session_factory):
    """A run that stopped before packaging still produced real results."""
    RunLedger(run_state, session_factory=session_factory)
    solve_dir = Path(run_state.run_dir) / "artifacts" / "solve"
    solve_dir.mkdir(parents=True)

    (solve_dir / "outcome1.json").write_text(
        json.dumps({"status": "failed", "summary": {"statistics": {"stale": 1}}}),
        encoding="utf-8",
    )
    (solve_dir / "outcome2.json").write_text(
        json.dumps(
            {
                "run_id": run_state.run_id,
                "status": "success",
                "exit_code": 0,
                "runtime_seconds": 12.5,
                "execution_real": True,
                "output_dir": str(solve_dir / "attempt2"),
                "artifacts": ["result1.xlsx", "result2.xlsx"],
                "summary": {
                    "statistics": {"peak_remaining": 2.55},
                    "self_check": {"max_abs_error": 0.01},
                    "per_subproblem": {"problem1": {"n_steps": 1800}},
                },
            }
        ),
        encoding="utf-8",
    )
    os.utime(solve_dir / "outcome1.json", (1_600_000_000, 1_600_000_000))
    os.utime(solve_dir / "outcome2.json", (1_700_000_000, 1_700_000_000))

    run_state.begin("solve", "attempt 2: docker sandbox")
    run_state.complete("solve", "2 output file(s)")

    results = _rows(session_factory)[0].results
    assert results["source"] == "solve_outcome"
    assert results["attempt"] == 2
    assert results["result_files"] == ["result1.xlsx", "result2.xlsx"]
    assert results["statistics"] == {"peak_remaining": 2.55}
    assert results["execution_real"] is True
    # Solver detail stays out of the results record.
    assert "per_subproblem" not in results


def test_ledger_mirrors_the_domain_state(run_state, session_factory):
    ledger = RunLedger(run_state, session_factory=session_factory)

    domain = ProblemState(
        raw_problem="the problem statement", title="A", competition="CUMCM"
    )
    domain.metadata_ = {"analysis": {"subproblems": []}, "jury_result": {"winner": 1}}
    domain.results = {"verified": True}
    ledger.observe_domain_state(domain)

    run_state.begin("intake", "reading problem statement and attachments")

    row = _rows(session_factory)[0]
    assert row.title == "A"
    assert row.competition == "CUMCM"
    assert row.raw_problem == "the problem statement"
    assert row.metadata_["jury_result"] == {"winner": 1}
    # No manifest exists yet, so the object's own results are kept.
    assert row.results == {"verified": True}


def test_ledger_stores_a_domain_state_that_carries_datetimes(
    run_state, session_factory
):
    """A datetime in the domain metadata must not disable the ledger.

    The domain object is allowed to hold datetimes, UUIDs and enums; the JSON
    columns are not. This used to abort the write with "Object of type datetime
    is not JSON serializable" and switch the ledger off for the whole run.
    """
    from datetime import datetime, timezone
    from uuid import uuid4

    ledger = RunLedger(run_state, session_factory=session_factory)

    domain = ProblemState(raw_problem="the problem", title="A", competition="CUMCM")
    stamp = datetime(2026, 9, 26, 3, 34, 53, tzinfo=timezone.utc)
    domain.metadata_ = {
        "analysis": {"created_at": stamp, "analysis_id": uuid4()},
        "checks": [{"finished_at": stamp}],
    }
    ledger.observe_domain_state(domain)

    run_state.begin("intake", "reading")

    rows = _rows(session_factory)
    assert len(rows) == 1
    assert rows[0].metadata_["analysis"]["created_at"] == stamp.isoformat()
    assert not [note for note in run_state.notes if "run ledger" in note]


def test_unchanged_run_is_not_rewritten(run_state, session_factory):
    ledger = RunLedger(run_state, session_factory=session_factory)

    run_state.begin("intake", "reading")
    writes = ledger.writes

    run_state.notify()
    run_state.notify()

    assert ledger.writes == writes


# ── Failure handling ─────────────────────────────────────────


def test_unusable_database_does_not_break_the_run(run_state):
    class BrokenFactory:
        def __call__(self):
            raise RuntimeError("database is down")

    ledger = RunLedger(run_state, session_factory=BrokenFactory())

    # The run keeps working, and the file that resumes it is still written.
    run_state.begin("intake", "reading problem statement and attachments")
    run_state.complete("intake", "2 file(s)")

    assert ledger.disabled is True
    assert any("run ledger" in note for note in run_state.notes)
    reloaded = RunState.load(run_state.run_dir)
    assert reloaded.current_stage == "intake"
    assert reloaded.stages["intake"].status.value == "completed"


def test_ledger_stops_writing_after_it_disables_itself(run_state):
    attempts = []

    class FlakyFactory:
        def __call__(self):
            attempts.append(1)
            raise RuntimeError("database is down")

    ledger = RunLedger(run_state, session_factory=FlakyFactory())

    run_state.begin("intake", "reading")
    run_state.complete("intake", "2 file(s)")
    run_state.begin("understand", "ProblemAgent (staged)")

    assert ledger.disabled is True
    # One failure is enough: the ledger must not retry on every stage.
    assert len(attempts) == 1


# ── Wiring into a real run ───────────────────────────────────
#
# These drive the real pipeline rather than a hand-built RunState. With no
# input files the run stops at clarification without contacting a model, so
# the wiring is covered offline.


@pytest.fixture
def configured_database(monkeypatch):
    """Point the project's sync engine at a chosen URL for one test."""
    import mathmodel.autopilot.ledger as ledger_module
    from mathmodel import database as database_module

    def _configure(url: str) -> None:
        monkeypatch.setattr(database_module.settings, "database_url", url)
        monkeypatch.setattr(ledger_module, "_SCHEMA_CHECKED", False)
        database_module.reset_engines()

    yield _configure

    database_module.reset_engines()


async def test_a_real_run_writes_its_row(
    tmp_path, configured_database, monkeypatch
):
    from mathmodel.autopilot import CUMCMAutopilot

    url = f"sqlite:///{(tmp_path / 'ledger.db').as_posix()}"
    configured_database(url)

    result = await CUMCMAutopilot(workspace=tmp_path).run(problem_files=[])

    assert result.status == "needs_input"
    assert not [note for note in result.notes if "run ledger" in note]

    engine = create_engine(url)
    factory = sessionmaker(engine, expire_on_commit=False)
    try:
        rows = _rows(factory)
    finally:
        engine.dispose()

    assert len(rows) == 1
    row = rows[0]
    assert row.run_id == result.run_id
    # The run is waiting for the problem files, which is `pending`, not running.
    assert row.status == ProblemStateStatus.PENDING
    assert row.current_stage == ProblemStateStage.INGEST
    history = {entry["stage"]: entry for entry in row.stage_history}
    assert history["intake"]["status"] == "completed"
    assert history["clarify"]["problem_stage"] == "ingest"


async def test_a_real_run_survives_an_unreachable_database(
    tmp_path, configured_database
):
    from mathmodel.autopilot import CUMCMAutopilot

    # Nothing listens here, and the driver for this URL is absent in the test
    # environment: either way the ledger must give up rather than stall.
    configured_database("postgresql://nobody@127.0.0.1:1/none")

    result = await CUMCMAutopilot(workspace=tmp_path).run(problem_files=[])

    assert result.status == "needs_input"
    assert [note for note in result.notes if "run ledger" in note]
    # The file that resumes the run is unaffected. (The run directory is named
    # after a throwaway id, so it is taken from the result.)
    reloaded = RunState.load(result.run_dir)
    assert reloaded.stages["intake"].status.value == "completed"
