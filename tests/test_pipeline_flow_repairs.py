"""Flow defects in `CUMCMAutopilot._drive`, pinned by test.

Two were fixed here:

* the eligibility gate had no authority. Every candidate was stored as
  eligible with the reason "autopilot intake", so `ModelJury` scored
  candidates the gate exists to reject.
* the attempt number of a cached solver run was read from the last character
  of its directory name, which reports attempt 10 as `report0.json`, and the
  search for a verified attempt was bounded by the *current* repair limit, so
  lowering that limit silently orphaned an already verified run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from mathmodel.autopilot.pipeline import _report_artifact, _verified_attempt_numbers
from mathmodel.autopilot.state import RunState
from mathmodel.domain.analysis import ProblemAnalysis
from mathmodel.domain.candidates import ModelCandidate, ModelFamily

PIPELINE_SOURCE = Path(__file__).resolve().parents[1] / "src" / "mathmodel" / "autopilot" / "pipeline.py"


@pytest.fixture
def configured_database(monkeypatch):
    """Point the project's sync engine at a per-test sqlite file."""
    import mathmodel.autopilot.ledger as ledger_module
    from mathmodel import database as database_module

    def _configure(url: str) -> None:
        monkeypatch.setattr(database_module.settings, "database_url", url)
        monkeypatch.setattr(ledger_module, "_SCHEMA_CHECKED", False)
        database_module.reset_engines()

    yield _configure
    database_module.reset_engines()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_run(
    tmp_path: Path, candidates: list[ModelCandidate]
) -> tuple[str, dict[str, str]]:
    """Create a run that is already past intake, understanding and exploring."""
    from mathmodel.autopilot.intake import ProblemContext, build_clarification_questions

    run_id = "seeded-flow-run"
    run_dir = tmp_path / run_id
    state = RunState.create(run_dir)
    for stage in ("intake", "understand", "explore"):
        state.begin(stage, "seeded")
        state.complete(stage, "seeded")
    state.save()

    context = ProblemContext(
        problem_text="Dry a porous cylinder and report the drying time per subproblem.",
        problem_file_names=["problem.pdf"],
        required_outputs=["result1.xlsx"],
    )
    _write(
        run_dir / "artifacts" / "intake" / "problem_context.json",
        context.model_dump(mode="json"),
    )
    _write(
        run_dir / "artifacts" / "analysis" / "analysis.json",
        ProblemAnalysis(
            background="A porous cylinder is dried in a chamber of controlled air.",
            core_problem="Predict the drying time and the moisture profile.",
        ).model_dump(mode="json"),
    )
    _write(
        run_dir / "artifacts" / "models" / "candidates.json",
        [c.model_dump(mode="json") for c in candidates],
    )
    # Whatever the intake considers blocking is answered, so this test reaches
    # the model-selection stage instead of stopping at clarification.
    answers = {
        question.question_id: "not stated in the problem statement"
        for question in build_clarification_questions(context)
        if question.blocking
    }
    return run_id, answers


async def test_a_candidate_that_violates_a_hard_constraint_is_never_scored(
    tmp_path, configured_database, monkeypatch
):
    from mathmodel.agents.model_jury import ModelJury
    from mathmodel.autopilot import CUMCMAutopilot

    configured_database(f"sqlite:///{(tmp_path / 'flow.db').as_posix()}")

    candidates = [
        ModelCandidate(
            name="Planar diffusion with a measured humidity history",
            model_family=ModelFamily.SIMULATION,
            summary="One-dimensional drying model driven by the chamber series.",
            risk_flags=[
                "hard_constraint: the chamber humidity series the model needs "
                "was not supplied with the problem"
            ],
        )
    ]
    run_id, answers = _seed_run(tmp_path, candidates)

    def _jury_must_not_run(self, state):  # pragma: no cover - failure path
        raise AssertionError("ModelJury scored a candidate the gate rejected")

    monkeypatch.setattr(ModelJury, "run", _jury_must_not_run)

    result = await CUMCMAutopilot(workspace=tmp_path).run(
        problem_files=[], run_id=run_id, answers=answers
    )

    assert result.status == "blocked"
    assert "no candidate model passed eligibility" in result.blockers

    # The run record must show the gate failing the stage, not a silent skip.
    state = RunState.load(tmp_path / run_id)
    assert state.stage("select").status == "failed"
    assert "eligibility" in state.stage("select").detail.lower()


def test_no_candidate_is_assumed_eligible_any_more():
    """Guard against the placeholder being reintroduced."""
    source = PIPELINE_SOURCE.read_text(encoding="utf-8")

    assert "autopilot intake" not in source
    assert "EligibilityGate(" in source


def test_verified_attempts_are_read_from_disk_newest_first(tmp_path):
    verify = tmp_path / "verify"
    verify.mkdir()
    for name in ("report1.json", "report2.json", "report10.json", "summary.json"):
        (verify / name).write_text("{}", encoding="utf-8")

    assert _verified_attempt_numbers(verify) == [10, 2, 1]


def test_a_verified_attempt_above_the_repair_limit_is_still_found(tmp_path):
    """Lowering the attempt limit must not orphan an already verified run."""
    verify = tmp_path / "verify"
    verify.mkdir()
    (verify / "report6.json").write_text("{}", encoding="utf-8")

    assert _verified_attempt_numbers(verify) == [6]


def test_attempt_numbers_come_from_the_name_not_its_last_digit(tmp_path):
    verify_dir = tmp_path / "verify"
    solve_dir = tmp_path / "solve"

    assert _report_artifact(verify_dir, solve_dir / "attempt10") == [
        str(verify_dir / "report10.json")
    ]
    assert _report_artifact(verify_dir, solve_dir / "attempt3") == [
        str(verify_dir / "report3.json")
    ]
    assert _report_artifact(verify_dir, solve_dir / "latest") == []


def _figure_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _seed_publication(
    run_dir: Path,
    *,
    inputs: dict,
    figure_file: Optional[Path] = None,
    paper_done: bool = True,
) -> RunState:
    """Write the artifacts a finished publication leaves behind."""
    from mathmodel.autopilot.pipeline import _ArtifactPaths
    from mathmodel.documents import FigureRecord, TableRecord
    from mathmodel.paper import PaperIR

    artifacts = _ArtifactPaths(run_dir)
    figures = []
    if figure_file is not None:
        figures.append(
            FigureRecord(
                title="温度曲线",
                artifact_path=str(figure_file),
                hash=_figure_hash(figure_file),
                source_execution_ids=["EVD-EXECUTION"],
            ).model_dump(mode="json")
        )
    _write(artifacts.figures / "figures.json", figures)
    _write(
        artifacts.tables / "tables.json",
        [TableRecord(title="求解结果统计").model_dump(mode="json")],
    )
    _write(
        artifacts.paper / "paper_ir.json",
        PaperIR(title="多孔圆柱干燥模型", abstract="摘要正文。").model_dump(mode="json"),
    )
    _write(artifacts.paper / "publication_inputs.json", inputs)

    state = RunState.create(run_dir)
    for stage in ("intake", "understand", "explore", "select", "model"):
        state.begin(stage, "seeded")
        state.complete(stage, "seeded")
    if paper_done:
        state.begin("paper", "seeded")
        state.complete("paper", "seeded")
    state.save()
    return state


def test_a_finished_publication_is_reused_when_its_inputs_are_unchanged(tmp_path):
    from mathmodel.autopilot import CUMCMAutopilot
    from mathmodel.autopilot.pipeline import _ArtifactPaths

    run_dir = tmp_path / "reuse"
    figure_file = run_dir / "artifacts" / "figures" / "trend.png"
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    figure_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    inputs = {"outcome": "RUN-cached", "statistics": {"drying_time": 3.5}}
    state = _seed_publication(run_dir, inputs=inputs, figure_file=figure_file)

    reusable = CUMCMAutopilot(workspace=tmp_path)._reusable_publication(
        _ArtifactPaths(run_dir), state, inputs
    )

    assert reusable is not None
    figures, tables, paper = reusable
    assert [f.figure_id for f in figures.all()]  # the recorded ids survive
    assert paper.title == "多孔圆柱干燥模型"
    assert tables.all()


def test_a_publication_is_not_reused_when_the_verified_inputs_changed(tmp_path):
    from mathmodel.autopilot import CUMCMAutopilot
    from mathmodel.autopilot.pipeline import _ArtifactPaths

    run_dir = tmp_path / "changed"
    inputs = {"outcome": "RUN-cached", "statistics": {"drying_time": 3.5}}
    state = _seed_publication(run_dir, inputs=inputs)

    changed = {"outcome": "RUN-cached", "statistics": {"drying_time": 4.25}}

    assert (
        CUMCMAutopilot(workspace=tmp_path)._reusable_publication(
            _ArtifactPaths(run_dir), state, changed
        )
        is None
    )


def test_a_publication_whose_figure_disappeared_is_not_reused(tmp_path):
    from mathmodel.autopilot import CUMCMAutopilot
    from mathmodel.autopilot.pipeline import _ArtifactPaths

    run_dir = tmp_path / "vanished"
    figure_file = run_dir / "artifacts" / "figures" / "trend.png"
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    figure_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    inputs = {"outcome": "RUN-cached"}
    state = _seed_publication(run_dir, inputs=inputs, figure_file=figure_file)
    figure_file.unlink()

    assert (
        CUMCMAutopilot(workspace=tmp_path)._reusable_publication(
            _ArtifactPaths(run_dir), state, inputs
        )
        is None
    )


def test_an_unfinished_paper_is_never_reused(tmp_path):
    from mathmodel.autopilot import CUMCMAutopilot
    from mathmodel.autopilot.pipeline import _ArtifactPaths

    run_dir = tmp_path / "unfinished"
    inputs = {"outcome": "RUN-cached"}
    state = _seed_publication(run_dir, inputs=inputs, paper_done=False)

    assert (
        CUMCMAutopilot(workspace=tmp_path)._reusable_publication(
            _ArtifactPaths(run_dir), state, inputs
        )
        is None
    )
