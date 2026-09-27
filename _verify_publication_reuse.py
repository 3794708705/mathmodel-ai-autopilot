"""Independent check: resuming a finished run must not rewrite its paper.

A unit test can only show that `_reusable_publication` returns the right thing.
This drives the real `_drive` over a **real completed run** copied out of
`runs/` (that directory is untracked, which is why this is a script and not a
test): the front half resumes from its own cached artifacts, and the paper
writer, the section writer and the figure proposer are all replaced by stubs
that fail loudly if the publication is regenerated.

It then proves the other direction: with the model version changed, the stored
publication no longer matches the verified inputs, so the writer *is* called.

Run from the project root:
    .\\.venv\\Scripts\\python.exe _verify_publication_reuse.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

SEED = ROOT / "runs" / "reliability" / "c03"

outcomes: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    outcomes.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _verified_attempt(verify_dir: Path) -> tuple[str, Path]:
    """The highest-numbered attempt whose report says PASS."""
    passing = []
    for report in verify_dir.glob("report*.json"):
        if str(_read(report).get("overall", "")).upper() == "PASS":
            passing.append(int(report.stem.removeprefix("report")))
    if not passing:
        raise SystemExit("the seed run has no passing verification report")
    attempt = max(passing)
    return str(attempt), verify_dir / f"report{attempt}.json"


async def main() -> int:
    if not SEED.exists():
        raise SystemExit(f"seed run is missing: {SEED}")

    from mathmodel import database as database_module
    from mathmodel.autopilot import CUMCMAutopilot, deliver, pipeline
    from mathmodel.autopilot.intake import ProblemContext
    from mathmodel.autopilot.state import RunState
    from mathmodel.paper import PaperIR

    workdir = Path(tempfile.mkdtemp(prefix="publication-reuse-"))
    run_dir = workdir / SEED.name
    shutil.copytree(SEED, run_dir)
    # `RunState.load` keeps the `run_dir` recorded in the state file, so the
    # copy must be retargeted or the resume would write into the archive.
    copied = RunState.load(run_dir)
    copied.run_dir = str(run_dir)
    copied.save()

    url = f"sqlite:///{(workdir / 'verify.db').as_posix()}"
    database_module.settings.database_url = url
    database_module.reset_engines()
    import mathmodel.autopilot.ledger as ledger_module

    ledger_module._SCHEMA_CHECKED = False

    artifacts = pipeline._ArtifactPaths(run_dir)
    attempt, _ = _verified_attempt(artifacts.verify)
    solve_dir = artifacts.solve / f"attempt{attempt}"
    outcome = _read(artifacts.solve / f"outcome{attempt}.json")
    model_payload = _read(artifacts.model / "math_model.json")
    analysis_payload = _read(artifacts.analysis / "analysis.json")
    context = ProblemContext.model_validate(_read(artifacts.intake / "problem_context.json"))

    autopilot = CUMCMAutopilot(workspace=workdir)
    # The stored publication is keyed on the verified inputs; this run predates
    # that fingerprint file, so it is written here exactly as a run that has
    # just finished with this code would have written it — through the same
    # validated models `_drive` holds, not from raw JSON slices.
    from mathmodel.autopilot.codegen import ExecutionOutcome
    from mathmodel.domain.analysis import ProblemAnalysis
    from mathmodel.domain.math_model import MathematicalModel

    publication_inputs = {
        "outcome": ExecutionOutcome.model_validate(outcome).run_id,
        "model_version": MathematicalModel.model_validate(model_payload).version,
        "statistics": (outcome.get("summary") or {}).get("statistics") or {},
        "subproblems": [
            sub.model_dump(mode="json")
            for sub in ProblemAnalysis.model_validate(analysis_payload).subproblems
        ],
        "output_summaries": autopilot._output_summaries(
            solve_dir, context.required_outputs
        ),
    }
    _write(artifacts.paper / "publication_inputs.json", publication_inputs)

    before = PaperIR.model_validate(_read(artifacts.paper / "paper_ir.json"))

    calls = {"outline": 0, "section": 0, "abstract": 0, "propose": 0}

    def _forbidden(name: str):
        async def _call(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} was called although the publication is reusable")

        return _call

    def _stub_pdf(tex_source, build_dir, image=None, timeout=None):
        pdf = Path(build_dir) / "paper.pdf"
        if not pdf.exists():
            pdf.write_bytes(b"%PDF-1.4\n% stub\n")
        return deliver.PdfBuildResult(
            success=True,
            pdf_path=str(pdf),
            pdf_size=pdf.stat().st_size,
            tex_path=str(Path(build_dir) / "paper.tex"),
            command="stub",
            exit_code=0,
            compiler_available=True,
            pdf_header_ok=True,
        )

    with patch.object(deliver.PaperWriter, "build_outline", _forbidden("outline")), patch.object(
        deliver.PaperWriter, "write_section", _forbidden("section")
    ), patch.object(
        deliver.PaperWriter, "write_abstract", _forbidden("abstract")
    ), patch.object(
        deliver.FigureBuilder, "propose", _forbidden("propose")
    ), patch.object(
        pipeline, "build_pdf", _stub_pdf
    ):
        result = await autopilot.resume(run_dir)

    state = RunState.load(run_dir)
    reused_stages = {
        stage: state.stage(stage).detail
        for stage in ("figures", "tables", "paper")
    }
    print(f"\nresumed run status: {result.status}")
    for stage, detail in reused_stages.items():
        print(f"  {stage}: {state.stage(stage).status} — {detail}")
    if result.blockers:
        print(f"  blockers: {result.blockers}")
    if result.notes:
        print(f"  notes: {result.notes}")
    print()

    check(
        "no model call was made while resuming a finished publication",
        sum(calls.values()) == 0,
        f"writer/proposer calls: {calls}",
    )
    check(
        "the publication stages report a reuse",
        all("reused" in (detail or "") for detail in reused_stages.values()),
        "; ".join(f"{k}={v}" for k, v in reused_stages.items()),
    )
    after = PaperIR.model_validate(_read(artifacts.paper / "paper_ir.json"))
    check(
        "the paper on disk is the one that was already delivered",
        after.paper_id == before.paper_id
        and [s.title for s in after.sections] == [s.title for s in before.sections],
        f"{len(after.sections)} section(s), paper_id unchanged: {after.paper_id == before.paper_id}",
    )
    check(
        "the resumed run still reaches a finished package",
        result.status == "completed",
        f"status={result.status}",
    )

    # Direction two: a changed verified solution must not be described by the
    # stored paper.
    calls.clear()
    calls.update({"outline": 0, "section": 0, "abstract": 0, "propose": 0})
    model_payload["version"] = int(model_payload.get("version") or 1) + 1
    _write(artifacts.model / "math_model.json", model_payload)

    regenerated = False
    with patch.object(deliver.PaperWriter, "build_outline", _forbidden("outline")), patch.object(
        deliver.FigureBuilder, "propose", _forbidden("propose")
    ):
        try:
            await autopilot.resume(run_dir)
        except Exception as exc:  # the stubs raise on purpose
            regenerated = True
            print(f"\nsecond resume stopped at the stub as expected: {type(exc).__name__}")
    check(
        "a changed model version forces the publication to be written again",
        regenerated and sum(calls.values()) > 0,
        f"writer/proposer calls: {calls}",
    )

    failed = [name for name, ok, _ in outcomes if not ok]
    print(f"\n{len(outcomes) - len(failed)}/{len(outcomes)} checks passed")
    if failed:
        print("failed: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
