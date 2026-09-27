"""MathModel AI — CUMCM Autopilot: the continuous workflow.

Upload problem + attachments, then run one continuous path:

    intake → clarify → understand → registries → explore → select → model
      → codegen → solve → verify ─(FAIL)→ codegen (repair)
                              └(PASS)→ evidence → figures → tables → paper
                                      → audit → pdf → package → final check

Verification is a hard gate: nothing unverified reaches the paper.
State is persisted after every stage so a run can be resumed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from mathmodel.agents.base import AgentStatus
from mathmodel.agents.eligibility_gate import EligibilityGate
from mathmodel.agents.math_modeler import MathModeler
from mathmodel.agents.model_explorer import ModelExplorer
from mathmodel.agents.model_jury import ModelJury
from mathmodel.agents.problem_agent import ProblemAgent
from mathmodel.autopilot.codegen import (
    ExecutionOutcome,
    GeneratedProgram,
    SandboxProgramRunner,
    SolverCodeGenerator,
)
from mathmodel.autopilot.deliver import (
    DATA_STATUS_FORMAL,
    FigureBuilder,
    PaperContextPackage,
    PaperWriter,
    SupportPackageBuilder,
    TableBuilder,
    build_cumcm_latex,
    build_pdf,
    drafts_to_paper_ir,
)
from mathmodel.autopilot.intake import ProblemContext, ProblemIntake, build_clarification_questions
from mathmodel.autopilot.ledger import RunLedger
from mathmodel.autopilot.state import (
    ClarificationQuestion,
    RunState,
    RunStatus,
    StageStatus,
)
from mathmodel.autopilot.verify import (
    CheckStatus,
    DeterministicVerifier,
    IndependentVerifier,
    VerificationCheck,
    VerificationReport,
    build_report,
    checks_from_independent_summary,
    compare_statistics,
    describe_solution_files,
    model_integrates_differential_equations,
    unevidenced_by_diverged_recomputation,
    unaccountable_formula_verdict,
)
from mathmodel.domain.candidates import ModelCandidate
from mathmodel.domain.eligibility import EligibilityPolicy
from mathmodel.domain.math_model import MathematicalModel
from mathmodel.domain.state_helpers import (
    load_analysis,
    load_candidates,
    load_eligibility_results,
    load_jury_result,
    store_analysis,
    store_candidates,
    store_jury_result,
)
from mathmodel.domain.verification import GateStatus
from mathmodel.evidence import (
    Claim,
    ClaimStatus,
    ClaimType,
    EvidenceRef,
    EvidenceSourceType,
    EvidenceStore,
)
from mathmodel.models.problem_state import ProblemState
from mathmodel.documents import (
    FigureRecord,
    FigureRegistry,
    TableRecord,
    TableRegistry,
)
from mathmodel.routing.router import ModelRouter
from mathmodel.submission import CompetitionProfile, SubmissionCheckAgent, SubmissionStatus

MAX_REPAIR_ATTEMPTS = 3
MAX_AGENT_ATTEMPTS = 3
# The independent verifier is generated code too; a crash in it is a bug to
# repair, not evidence about the solution under test.
MAX_VERIFIER_ATTEMPTS = 3

logger = logging.getLogger(__name__)


class AutopilotResult(BaseModel):
    run_id: str
    status: str
    run_dir: str
    output_dir: str = ""
    paper_pdf: str = ""
    verification: str = ""
    pending_questions: list[ClarificationQuestion] = Field(default_factory=list)
    stage_summary: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _verified_attempt_numbers(verify_dir: Path) -> list[int]:
    """Attempt numbers that have a verification report, newest first.

    Read from disk rather than derived from the current attempt limit: a run
    verified under a larger limit must stay reusable after that limit is
    lowered, and raising the limit must not skip an attempt already on disk.
    """
    numbers: list[int] = []
    for path in verify_dir.glob("report*.json"):
        match = re.fullmatch(r"report(\d+)\.json", path.name)
        if match is not None:
            numbers.append(int(match.group(1)))
    return sorted(numbers, reverse=True)


def _report_artifact(verify_dir: Path, solve_dir: Path) -> list[str]:
    """The verification report that belongs to a cached solver attempt.

    The attempt number is read out of the `attempt<N>` artifact directory
    instead of its last character: attempt 10 would otherwise be recorded as
    `report0.json`.
    """
    match = re.fullmatch(r"attempt(\d+)", solve_dir.name)
    if match is None:
        return []
    return [str(verify_dir / f"report{match.group(1)}.json")]


class CUMCMAutopilot:
    """Drives one problem from upload to a verified paper and support package."""

    def __init__(
        self,
        workspace: str | Path = "runs",
        sandbox_image: str = "mathmodel-ai-autopilot:latest",
        paper_image: str = "mathmodel-ai-paper:phase6",
        max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    ):
        self._workspace = Path(workspace)
        self._sandbox_image = sandbox_image
        self._paper_image = paper_image
        self._max_repair = max_repair_attempts
        self._router = ModelRouter()
        self._runner = SandboxProgramRunner(image=sandbox_image)

    # ══════════════════════════════════════════════════════════
    # Entry points
    # ══════════════════════════════════════════════════════════

    async def run(
        self,
        problem_files: list[str | Path],
        competition: str = "CUMCM",
        run_id: Optional[str] = None,
        answers: Optional[dict[str, str]] = None,
    ) -> AutopilotResult:
        run_dir = self._workspace / (run_id or RunState().run_id)
        if run_id and (run_dir / "pipeline_state.json").exists():
            # Re-entering a known run resumes it instead of discarding progress.
            state = RunState.load(run_dir)
            state.status = RunStatus.RUNNING
            state.pending_questions = []
            if answers:
                state.answers.update(answers)
            state.save()
        else:
            state = RunState.create(run_dir)
            state.answers = dict(answers or {})
            state.save()
        return await self._drive(state, problem_files)

    async def resume(
        self,
        run_dir: str | Path,
        answers: Optional[dict[str, str]] = None,
    ) -> AutopilotResult:
        state = RunState.load(run_dir)
        if answers:
            state.answers.update(answers)
        state.pending_questions = []
        state.status = RunStatus.RUNNING
        state.save()
        return await self._drive(state, state.input_files)

    def answer(self, run_dir: str | Path, answers: dict[str, str]) -> None:
        """Record user answers so a paused run can continue."""
        state = RunState.load(run_dir)
        state.answers.update(answers)
        state.save()

    # ══════════════════════════════════════════════════════════
    # Drive
    # ══════════════════════════════════════════════════════════

    @staticmethod
    async def _retry_call(stage: str, call, max_attempts: int = MAX_AGENT_ATTEMPTS):
        """Run one plain LLM call, retrying transient provider failures.

        The local gateway and remote APIs both drop connections occasionally.
        Losing an entire multi-hour run to one reset socket is not acceptable
        when the work already completed is on disk and the call is idempotent.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await call()
            except Exception as exc:  # noqa: BLE001 - provider errors vary widely
                last_exc = exc
                logger.warning("%s attempt %d failed: %s", stage, attempt, exc)
                if attempt < max_attempts:
                    await asyncio.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"{stage} failed after {max_attempts} attempts: {last_exc}")

    @staticmethod
    def _verifiable_extras(report, output_summaries: dict[str, Any]) -> set[float]:
        """Numbers a paper may legitimately cite besides the solver statistics.

        Row counts of the produced result files and anything the verification
        stage reported are real, checkable facts, so a paper that quotes them is
        not fabricating.
        """
        extras = _verification_numbers(report)
        _collect_numbers(output_summaries, extras)
        return extras

    async def _repair_model(
        self,
        *,
        state: RunState,
        state_obj,
        artifacts: "_ArtifactPaths",
        problem_context: ProblemContext,
        feedback: list[str],
    ) -> Optional[MathematicalModel]:
        """Re-formalize the model using verification's disagreement as feedback."""
        state.begin("model", "MathModeler (repair after verification failure)")
        logger.warning(
            "Repairing the mathematical model; verification disagreed: %s",
            "; ".join(feedback)[:300],
        )
        try:
            result = await self._attempt(
                "model-repair",
                lambda: MathModeler(router=self._router).run(
                    state_obj, external_feedback=feedback
                ),
                max_attempts=2,
            )
        except Exception as exc:
            logger.warning("Model repair failed: %s", exc)
            state.complete("model", "repair failed; keeping the previous model", [])
            return None
        if result is None or result.status != AgentStatus.COMPLETED:
            state.complete("model", "repair failed; keeping the previous model", [])
            return None
        raw_model = (state_obj.metadata_ or {}).get("mathematical_model")
        if not raw_model:
            state.complete("model", "repair produced no model; keeping the previous one", [])
            return None
        try:
            repaired = MathematicalModel.model_validate(raw_model)
        except Exception as exc:
            logger.warning("Repaired model failed validation: %s", exc)
            state.complete("model", "repair produced an invalid model", [])
            return None
        _write_json(
            artifacts.model / "math_model.repaired.json",
            repaired.model_dump(mode="json"),
        )
        _write_json(artifacts.model / "math_model.json", repaired.model_dump(mode="json"))
        state.complete(
            "model",
            f"repaired: {repaired.name}: {len(repaired.variables)} var, "
            f"{len(repaired.constraints)} con, {len(repaired.equations)} eq",
            [str(artifacts.model / "math_model.json")],
        )
        return repaired

    def _load_verified_attempt(
        self,
        artifacts: "_ArtifactPaths",
        problem_context: ProblemContext,
        model: MathematicalModel,
    ) -> Optional[tuple[GeneratedProgram, Any, VerificationReport, Path]]:
        """Reload a solver run whose verification already passed.

        Resuming must not regenerate and re-execute a solution that is already
        verified: that burns the model budget and can replace a verified answer
        with a different, unverified one.
        """
        for attempt in _verified_attempt_numbers(artifacts.verify):
            report_path = artifacts.verify / f"report{attempt}.json"
            solver_path = artifacts.code / f"attempt{attempt}" / "solver.py"
            outcome_path = artifacts.solve / f"outcome{attempt}.json"
            solve_dir = artifacts.solve / f"attempt{attempt}"
            if not (report_path.exists() and solver_path.exists() and outcome_path.exists()):
                continue
            try:
                report_data = json.loads(report_path.read_text(encoding="utf-8"))
                if str(report_data.get("overall", "")).upper() != CheckStatus.PASS.value:
                    continue
                outcome = ExecutionOutcome.model_validate(
                    json.loads(outcome_path.read_text(encoding="utf-8"))
                )
                report = VerificationReport.model_validate(report_data)
            except Exception as exc:
                logger.warning("Cached verification in %s is unusable: %s", report_path, exc)
                continue
            if not outcome.succeeded or not solve_dir.is_dir():
                continue
            missing = [
                name for name in problem_context.required_outputs
                if not (solve_dir / name).exists()
            ]
            if missing:
                continue

            meta = {}
            meta_path = artifacts.code / f"attempt{attempt}" / "program.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            # The cached report was produced by whatever checks existed then.
            # Re-run the deterministic host checks, which cost nothing, so a
            # stale PASS cannot smuggle a result file past a check that was
            # added or tightened since.
            template_headers = self._template_headers(artifacts, problem_context)
            deterministic = DeterministicVerifier(artifacts.processed_data).run(
                outcome=outcome,
                output_dir=solve_dir,
                required_outputs=problem_context.required_outputs,
                template_headers=template_headers,
                expected_row_counts=self._expected_row_counts(
                    problem_context, template_headers
                ),
                template_sheets=self._template_sheets(problem_context),
                problem_text=problem_context.problem_text,
                requires_scheme_validation=model_integrates_differential_equations(
                    model
                ),
            )
            failed = [c for c in deterministic if c.status == CheckStatus.FAIL]
            if failed:
                logger.warning(
                    "Cached solver run %s no longer passes the deterministic "
                    "checks: %s",
                    solve_dir.name,
                    "; ".join(f"{c.name}: {c.detail}"[:200] for c in failed),
                )
                continue

            program = GeneratedProgram(
                approach=meta.get("approach", "") or "reused from previous run",
                code=solver_path.read_text(encoding="utf-8"),
                dependencies=meta.get("dependencies", []) or [],
                expected_outputs=meta.get("expected_outputs", []) or [],
            )
            return program, outcome, report, solve_dir
        return None

    def _reusable_publication(
        self,
        artifacts: "_ArtifactPaths",
        state: RunState,
        publication_inputs: dict[str, Any],
    ) -> Optional[tuple[FigureRegistry, TableRegistry, Any]]:
        """Reload the figures, tables and paper of an already written run.

        A resumed run must not re-draft a delivered paper, and it must not
        describe a changed solution with an old one either: the publication is
        reused only when the verified inputs it was written from are unchanged
        and every figure it refers to is still on disk. Anything else falls
        back to writing the publication again.
        """
        if not state.is_done("paper"):
            return None
        stored = self._load_json(artifacts.paper / "publication_inputs.json")
        if stored is None or _json_key(stored) != _json_key(publication_inputs):
            return None

        figure_payloads = self._load_json(artifacts.figures / "figures.json")
        table_payloads = self._load_json(artifacts.tables / "tables.json")
        paper_payload = self._load_json(artifacts.paper / "paper_ir.json")
        if figure_payloads is None or table_payloads is None or paper_payload is None:
            return None

        from mathmodel.paper import PaperIR

        try:
            figures = FigureRegistry()
            for payload in figure_payloads:
                figures.register(FigureRecord.model_validate(payload))
            tables = TableRegistry()
            for payload in table_payloads:
                tables.register(TableRecord.model_validate(payload))
            paper = PaperIR.model_validate(paper_payload)
        except Exception as exc:
            logger.warning(
                "The published artifacts of run %s cannot be reused: %s",
                state.run_id,
                exc,
            )
            return None

        missing = [
            figure.artifact_path
            for figure in figures.all()
            if figure.artifact_path and not Path(figure.artifact_path).exists()
        ]
        if missing:
            logger.warning(
                "Published figures of run %s are no longer on disk: %s",
                state.run_id,
                "; ".join(missing[:5]),
            )
            return None
        return figures, tables, paper

    async def _drive(
        self,
        state: RunState,
        problem_files: list[str | Path],
    ) -> AutopilotResult:
        run_dir = Path(state.run_dir)
        artifacts = _ArtifactPaths(run_dir)

        # The database mirrors this run; `pipeline_state.json` above stays the
        # mechanism that resumes it. A run that cannot reach a database still
        # runs: the ledger disables itself and says so in the notes.
        ledger = RunLedger(state)

        # Notes describe this attempt. Carrying them across a resume reports
        # problems the current attempt may already have fixed.
        if state.notes:
            state.notes = []

        if problem_files:
            state.input_files = [str(Path(p)) for p in problem_files]
            state.save()

        # ── 1. Intake ─────────────────────────────────────────
        context = self._load_json(artifacts.intake / "problem_context.json")
        if context is None or not state.is_done("intake"):
            state.begin("intake", "reading problem statement and attachments")
            try:
                intake = ProblemIntake(run_dir / "input")
                problem_context, records = intake.ingest(state.input_files)
            except Exception as exc:
                state.fail("intake", str(exc))
                return self._result(state, blockers=[f"intake failed: {exc}"])
            _write_json(artifacts.intake / "problem_context.json", problem_context.model_dump(mode="json"))
            _write_json(
                artifacts.intake / "files.json",
                [r.model_dump(mode="json") for r in records],
            )
            processed = intake.export_processed_data(problem_context, artifacts.processed_data)
            _write_json(artifacts.intake / "processed_data.json", processed)
            state.complete(
                "intake",
                f"{len(problem_context.attachments)} file(s), "
                f"problem text {len(problem_context.problem_text)} chars, "
                f"{len(processed)} processed data file(s)",
                [str(artifacts.intake / "problem_context.json")],
            )
        else:
            problem_context = ProblemContext.model_validate(context)

        # ── 2. Clarification ──────────────────────────────────
        # The stage is begun before it can stop for input, so a run waiting on
        # clarification still records where it is waiting.
        state.begin("clarify", "checking whether the upload is sufficient")
        questions = build_clarification_questions(problem_context)
        unanswered = [
            q for q in questions
            if q.blocking and q.question_id not in state.answers
        ]
        if unanswered:
            state.pending_questions = questions
            state.status = RunStatus.NEEDS_INPUT
            state.complete(
                "clarify",
                f"{len(unanswered)} blocking question(s) awaiting an answer",
            )
            return self._result(state, blockers=[q.question for q in unanswered])
        state.complete(
            "clarify",
            f"{len(questions)} question(s), all resolved"
            if questions else "upload is sufficient — no questions needed",
        )

        # ── 3. Problem understanding ──────────────────────────
        state_obj = ProblemState(
            raw_problem=self._agent_problem_text(problem_context),
            title=Path(problem_context.problem_file_names[0]).stem if problem_context.problem_file_names else "",
            competition=competition_name(problem_context),
        )
        # From here on the ledger mirrors this object's reasoning into the row.
        ledger.observe_domain_state(state_obj)
        analysis = None
        if state.is_done("understand"):
            cached = self._load_json(artifacts.analysis / "analysis.json")
            if cached:
                from mathmodel.domain.analysis import ProblemAnalysis
                analysis = ProblemAnalysis.model_validate(cached)
                # Downstream agents read the analysis from the problem state,
                # so a resumed run must restore it there too.
                store_analysis(state_obj, analysis)

        if analysis is None:
            state.begin("understand", "ProblemAgent (staged)")
            agent_result = await self._attempt(
                "understand",
                lambda: ProblemAgent(router=self._router)._run_staged(state_obj),
            )
            if agent_result is None or agent_result.status != AgentStatus.COMPLETED:
                detail = (
                    "; ".join(e.message for e in agent_result.errors)[:500]
                    if agent_result else "no result"
                )
                state.fail("understand", detail or "failed")
                return self._result(state, blockers=[f"problem understanding failed: {detail}"])
            analysis = load_analysis(state_obj)
            if analysis is None:
                state.fail("understand", "ProblemAnalysis not stored")
                return self._result(state, blockers=["problem understanding produced no analysis"])
            _write_json(artifacts.analysis / "analysis.json", analysis.model_dump(mode="json"))
            state.complete(
                "understand",
                f"{len(analysis.subproblems)} subproblem(s), "
                f"{len(analysis.evidence)} evidence item(s), "
                f"{len(analysis.ambiguities)} ambiguity(ies)",
                [str(artifacts.analysis / "analysis.json")],
            )
        else:
            state.complete("understand", "loaded from previous run", [])

        # ── 4. Registries: ambiguity / assumptions / data schema ──
        state.begin("registries", "ambiguity register, assumption ledger, data schema")
        ambiguity_register = self._ambiguity_register(analysis)
        assumption_ledger = self._assumption_ledger(analysis)
        data_schema = self._data_schema(problem_context)
        _write_json(artifacts.analysis / "ambiguity_register.json", ambiguity_register)
        _write_json(artifacts.analysis / "assumption_ledger.json", assumption_ledger)
        _write_json(artifacts.data / "data_schema.json", data_schema)
        state.complete(
            "registries",
            f"{len(ambiguity_register['ambiguities'])} ambiguities, "
            f"{len(assumption_ledger['assumptions'])} assumptions, "
            f"{len(data_schema['tables'])} data table(s)",
        )

        # ── 5. Candidate models ───────────────────────────────
        candidates: list[ModelCandidate] = []
        cached_candidates = self._load_json(artifacts.models / "candidates.json")
        if state.is_done("explore") and cached_candidates:
            candidates = [ModelCandidate.model_validate(c) for c in cached_candidates]
            store_candidates(state_obj, candidates)
        else:
            state.begin("explore", "ModelExplorer (staged)")
            explore_result = await self._attempt(
                "explore",
                lambda: ModelExplorer(router=self._router)._run_staged(state_obj),
            )
            if explore_result is None or explore_result.status != AgentStatus.COMPLETED:
                detail = (
                    "; ".join(e.message for e in explore_result.errors)[:500]
                    if explore_result else "no result"
                )
                state.fail("explore", detail or "failed")
                return self._result(state, blockers=[f"candidate generation failed: {detail}"])
            candidates = load_candidates(state_obj)
            if not candidates:
                state.fail("explore", "no candidates produced")
                return self._result(state, blockers=["no candidate models produced"])
            _write_json(
                artifacts.models / "candidates.json",
                [c.model_dump(mode="json") for c in candidates],
            )
            state.complete(
                "explore",
                f"{len(candidates)} candidate model(s)",
                [str(artifacts.models / "candidates.json")],
            )

        # ── 6. Model selection audit ──────────────────────────
        cached_jury = self._load_json(artifacts.models / "jury_result.json")
        if state.is_done("select") and cached_jury:
            from mathmodel.domain.jury import ModelJuryResult
            jury_result = ModelJuryResult.model_validate(cached_jury)
            store_jury_result(state_obj, jury_result)
            state.complete("select", "loaded from previous run", [])
        else:
            state.begin("select", "EligibilityGate + ModelJury")
            # The gate decides which candidates the jury is allowed to score.
            # It is a deterministic check over the facts the explorer recorded
            # (required data, plausibility, hard-constraint risks), so it is
            # executed here rather than every candidate being assumed eligible.
            gate_result = await EligibilityGate(
                policy=EligibilityPolicy()
            ).run(state_obj)
            if gate_result.status != AgentStatus.COMPLETED:
                detail = (
                    "; ".join(e.message for e in gate_result.errors)[:500] or "failed"
                )
                state.fail("select", f"EligibilityGate: {detail}")
                return self._result(
                    state, blockers=[f"eligibility checking failed: {detail}"]
                )
            eligibility = load_eligibility_results(state_obj)
            eligible = [r for r in eligibility if r.eligible]
            if not eligible:
                state.fail("select", "every candidate model failed eligibility")
                return self._result(
                    state, blockers=["no candidate model passed eligibility"]
                )

            jury = ModelJury(router=self._router)
            await jury.run(state_obj)
            jury_result = load_jury_result(state_obj)
            if jury_result is None or not jury_result.selected_model:
                state.fail("select", "ModelJury produced no selection")
                return self._result(state, blockers=["model selection failed"])
            _write_json(
                artifacts.models / "jury_result.json",
                jury_result.model_dump(mode="json"),
            )
            selection_audit = self._selection_audit(candidates, jury_result)
            _write_json(artifacts.models / "model_selection_audit.json", selection_audit)
            state.complete(
                "select",
                f"selected {jury_result.selected_model}"
                + (f", backup {jury_result.backup_model}" if jury_result.backup_model else "")
                + f" — {len(eligible)}/{len(eligibility)} candidate(s) eligible",
                [str(artifacts.models / "model_selection_audit.json")],
            )

        # ── 7. Mathematical model ─────────────────────────────
        cached_model = self._load_json(artifacts.model / "math_model.json")
        if state.is_done("model") and cached_model:
            model = MathematicalModel.model_validate(cached_model)
            state.complete("model", "loaded from previous run", [])
        else:
            state.begin("model", "MathModeler")
            model_result = await self._attempt(
                "model",
                lambda: MathModeler(router=self._router).run(state_obj),
                # MathModeler already retries internally with corrective
                # feedback, so only a couple of outer attempts are useful.
                max_attempts=2,
            )
            if model_result is None or model_result.status != AgentStatus.COMPLETED:
                detail = (
                    "; ".join(e.message for e in model_result.errors)[:500]
                    if model_result else "no result"
                )
                state.fail("model", detail or "failed")
                return self._result(state, blockers=[f"model formalization failed: {detail}"])
            raw_model = (state_obj.metadata_ or {}).get("mathematical_model")
            if not raw_model:
                state.fail("model", "model not stored in state")
                return self._result(state, blockers=["model formalization produced no model"])
            model = MathematicalModel.model_validate(raw_model)
            _write_json(artifacts.model / "math_model.json", model.model_dump(mode="json"))
            state.complete(
                "model",
                f"{model.name}: {len(model.variables)} var, {len(model.constraints)} con, "
                f"{len(model.equations)} eq",
                [str(artifacts.model / "math_model.json")],
            )

        # ── 8-10. Code generation → solve → verify (with repair) ──
        input_files = self._sandbox_inputs(artifacts, problem_context)
        generator = SolverCodeGenerator(self._router)
        independent = IndependentVerifier(self._router, self._runner)

        program: Optional[GeneratedProgram] = None
        outcome = None
        report = None
        feedback: list[str] = []
        # The competition supplies a blank template per result file. Its header
        # row is part of the required answer, so the solver is told the exact
        # columns instead of inventing its own wording.
        template_headers = self._template_headers(artifacts, problem_context)
        # Feedback aimed at the VERIFIER, not the solver. A verifier that ran but
        # did not honour its output contract is not the solver's fault, so this
        # is carried into the next attempt's verifier generation instead of
        # being blamed on the solver program.
        carried_verifier_feedback: list[str] = []
        # How many times each failure class has already been handed to the repair
        # loop. A repair loop that keeps seeing the same class is not converging,
        # and the useful thing to say then is not the error again but an
        # instruction to stop patching the site that reported it.
        failure_classes: dict[str, int] = {}
        # (attempt, self-check error / its own tolerance) for each attempt that
        # reached a usable solver run. Repairs that make a scheme WORSE are the
        # observed failure mode here: the same problem produced err=0.475 on one
        # attempt and err=57.319 on another, so a repair loop that always starts
        # from the latest attempt can walk away from a good discretisation and
        # never come back.
        attempt_error_ratios: list[tuple[int, float]] = []
        # The program text of every attempt that produced one, so that a repair
        # which is drifting AWAY from a better attempt can be based on that
        # attempt's code instead of on the worse one it just wrote. Asking the
        # model to restore the better attempt is only a request; keeping its
        # program here makes it possible to hand back.
        attempt_programs: dict[int, str] = {}
        attempt_failed_criteria: dict[int, list[str]] = {}

        for attempt in range(1, self._max_repair + 1):
            if attempt == 1:
                cached = self._load_verified_attempt(artifacts, problem_context, model)
                if cached is not None:
                    program, outcome, report, solve_dir = cached
                    logger.info(
                        "Resuming with the already verified solver run %s",
                        outcome.run_id,
                    )
                    state.complete(
                        "verify",
                        f"PASS (reused from previous run) — "
                        f"{len(report.checks)} check(s), 0 failures",
                        _report_artifact(artifacts.verify, solve_dir),
                    )
                    break
            state.begin(
                "codegen",
                f"attempt {attempt}/{self._max_repair}"
                + (" (repair)" if feedback else ""),
            )
            base_code, base_attempt = _repair_base_program(
                attempt_error_ratios,
                attempt_programs,
                attempt - 1,
                program.code if (program and feedback) else None,
            )
            if base_attempt is not None and feedback:
                feedback = [
                    f"NOTE: the program you are given is attempt {base_attempt}'s, "
                    f"NOT the one from the attempt that just failed. That earlier "
                    f"attempt scored better on the scheme check, so apply your fix "
                    f"to IT -- do not start from the most recent program."
                ] + list(feedback)
            try:
                program = await generator.generate(
                    model=model,
                    problem_text=problem_context.problem_text,
                    data_schema=json.dumps(data_schema, ensure_ascii=False, indent=2),
                    required_outputs=problem_context.required_outputs,
                    input_file_names=sorted(input_files),
                    previous_code=base_code,
                    failure_feedback=feedback or None,
                    output_headers=template_headers or None,
                )
            except Exception as exc:
                # A failed generation is not fatal while attempts remain: record
                # it and let the next attempt try again with the error as feedback.
                state.fail("codegen", str(exc))
                feedback = [
                    f"Program generation failed before any code ran: {exc}",
                    "Produce a complete, compact program that ends by printing the "
                    "required JSON summary line.",
                ]
                logger.warning("Code generation attempt %d failed: %s", attempt, exc)
                continue

            code_dir = artifacts.code / f"attempt{attempt}"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "solver.py").write_text(program.code, encoding="utf-8")
            _write_json(code_dir / "program.json", {
                "approach": program.approach,
                "dependencies": program.dependencies,
                "expected_outputs": program.expected_outputs,
            })
            state.complete(
                "codegen",
                f"attempt {attempt}: {len(program.code)} chars, approach: {program.approach[:120]}",
                [str(code_dir / "solver.py")],
            )

            # ── solve ─────────────────────────────────────────
            solve_dir = artifacts.solve / f"attempt{attempt}"
            state.begin("solve", f"attempt {attempt}: docker sandbox")
            outcome = await self._runner.run(program, input_files, solve_dir)
            _write_json(artifacts.solve / f"outcome{attempt}.json", outcome.model_dump(mode="json"))
            if not outcome.succeeded:
                state.fail(
                    "solve",
                    f"exit={outcome.exit_code} status={outcome.status}: "
                    f"{outcome.stderr.strip()[-300:]}",
                )
                feedback = [
                    f"Solver program failed to run: status={outcome.status} exit={outcome.exit_code}",
                    f"stderr tail: {outcome.stderr.strip()[-1500:]}",
                ]
                failure_class = _failure_class(outcome.stderr or "")
                seen = failure_classes.get(failure_class, 0)
                failure_classes[failure_class] = seen + 1
                if seen:
                    feedback.insert(
                        0,
                        _repeated_failure_note(failure_class, attempt, seen),
                    )
                continue
            state.complete(
                "solve",
                f"exit=0 runtime={outcome.runtime_seconds}s artifacts={len(outcome.artifacts)}",
                [str(p) for p in sorted(solve_dir.glob("*"))],
            )

            # ── verify ────────────────────────────────────────
            state.begin("verify", f"attempt {attempt}")
            template_headers = self._template_headers(artifacts, problem_context)
            deterministic = DeterministicVerifier(artifacts.processed_data).run(
                outcome=outcome,
                output_dir=solve_dir,
                required_outputs=problem_context.required_outputs,
                template_headers=template_headers,
                expected_row_counts=self._expected_row_counts(
                    problem_context, template_headers
                ),
                template_sheets=self._template_sheets(problem_context),
                problem_text=problem_context.problem_text,
                requires_scheme_validation=model_integrates_differential_equations(
                    model
                ),
            )

            checks = list(deterministic)
            solver_statistics = (outcome.summary or {}).get("statistics", {}) or {}
            # The verifier recomputes values, but it cannot check whether the solver
            # transcribed an empirical formula correctly, because it never sees the
            # solver's code and may make the same transcription slip itself. Asking
            # the solver to state the formulas it implemented lets the verifier
            # compare statement against code directly. This is the one defect class
            # that produces plausible-looking numbers, so no numeric comparison can
            # catch it.
            solver_formulas = (outcome.summary or {}).get("formulas", {}) or {}
            # The verifier is a separate program that has never seen the solver's
            # code, so it also has no idea how the output tables are laid out. It
            # guessed wrong on every run: filtering worksheets by name found
            # nothing on a template whose sheet is only `Sheet1`, and folding the
            # time column into the value range made every sheet look like it
            # spanned 0..t_end, so nothing could be classified and every derived
            # quantity came back None — which reads as the solver having produced
            # nothing. The real shape costs nothing to compute and removes the
            # guesswork.
            solution_structure = describe_solution_files(
                solve_dir, problem_context.required_outputs
            )
            independent_dir = artifacts.verify / f"attempt{attempt}"
            independent_dir.mkdir(parents=True, exist_ok=True)

            # The independent verifier is a separate layer, so it is skipped only
            # when there is genuinely nothing for it to inspect. Gating it on
            # "no deterministic check failed" meant a single formatting
            # complaint suppressed every other finding in the same run.
            outputs_readable = any(
                c.name == "required_outputs_present" and c.status == CheckStatus.PASS
                for c in deterministic
            )
            if outputs_readable:
                try:
                    verifier_inputs = dict(input_files)
                    for name in problem_context.required_outputs:
                        src = solve_dir / name
                        if src.exists():
                            import base64
                            verifier_inputs[f"output__{name}"] = base64.b64encode(
                                src.read_bytes()
                            ).decode("ascii")

                    verifier_program = None
                    verifier_outcome = None
                    verifier_feedback: list[str] = list(carried_verifier_feedback)
                    verifier_summary: dict[str, Any] = {}

                    for verifier_attempt in range(1, MAX_VERIFIER_ATTEMPTS + 1):
                        verifier_program = await independent.generate(
                            model=model,
                            problem_text=problem_context.problem_text,
                            required_outputs=problem_context.required_outputs,
                            solver_statistics=solver_statistics,
                            solver_formulas=solver_formulas,
                            available_inputs=sorted(input_files)
                            + [f"output/{n}" for n in problem_context.required_outputs],
                            failure_feedback=verifier_feedback or None,
                            data_schema=json.dumps(
                                data_schema, ensure_ascii=False, indent=2
                            ),
                            solution_structure=solution_structure or None,
                        )
                        (independent_dir / f"verifier{verifier_attempt}.py").write_text(
                            verifier_program.code, encoding="utf-8"
                        )
                        verifier_program_adapted = GeneratedProgram(
                            approach=verifier_program.approach,
                            code=_adapt_verifier_code(
                                verifier_program.code, problem_context.required_outputs
                            ),
                            dependencies=[],
                            expected_outputs=[],
                        )
                        verifier_outcome = await self._runner.run(
                            verifier_program_adapted,
                            verifier_inputs,
                            independent_dir / f"run{verifier_attempt}",
                        )
                        if verifier_outcome.succeeded:
                            break
                        # The verifier is itself generated code and can crash;
                        # hand its traceback back so it can be repaired.
                        verifier_feedback = [
                            "The verifier program exited with status "
                            f"{verifier_outcome.status} (exit={verifier_outcome.exit_code}).",
                            "stderr: " + ((verifier_outcome.stderr or "").strip()[-1200:] or "(empty)"),
                            "stdout tail: " + ((verifier_outcome.stdout or "").strip()[-400:] or "(empty)"),
                        ]
                        logger.warning(
                            "Independent verifier attempt %d crashed; retrying",
                            verifier_attempt,
                        )

                    if verifier_outcome is None or not verifier_outcome.succeeded:
                        detail = (
                            (verifier_outcome.stderr or "").strip()[-800:]
                            if verifier_outcome else "no execution recorded"
                        )
                        checks.append(VerificationCheck(
                            name="independent_verifier_executed",
                            category="independent",
                            status=CheckStatus.FAIL,
                            detail=f"Independent verifier failed to run: {detail}",
                        ))
                    else:
                        independent_checks, verifier_summary = checks_from_independent_summary(
                            verifier_outcome.summary or {},
                            required_outputs=problem_context.required_outputs,
                        )
                        checks.extend(independent_checks)
                        agreement = compare_statistics(
                            solver_statistics,
                            verifier_summary.get("statistics", {}) or {},
                        )
                        checks.append(agreement)
                        verifier_diverged = (
                            (agreement.evidence or {}).get("verifier_diverged") or []
                        )
                        # The verifier is generated code too, and it can contradict
                        # itself just as the solver can. A FAIL whose own
                        # per-formula verdicts all pass says the formulas were
                        # transcribed correctly, so the verdict is not evidence --
                        # reading it in place of the detail is the same mistake as
                        # trusting a solver's bare `passed` flag. Downgrade to
                        # NOT_RUN, which still fails the report (no standard is
                        # relaxed) but states the honest reason. Measured on the
                        # real A problem: a check failed with the detail
                        # "D_q1=e^-0.89/C ratio; D_q23/D_q4 use exp(-3850/T_K)
                        # Arrhenius ratio (T in Kelvin)" -- a statement that the
                        # formulas were correct -- and blocked the run on it.
                        checks = [
                            (
                                check.model_copy(
                                    update={
                                        "status": CheckStatus.NOT_RUN,
                                        "detail": (
                                            "Not verifiable: the verifier marked "
                                            "this check FAIL while every per-formula "
                                            "verdict it reported passes, so the "
                                            "verdict contradicts its own evidence. "
                                            f"Original finding: {unaccountable}"
                                        )[:2000],
                                    }
                                )
                                if (
                                    unaccountable := unaccountable_formula_verdict(check)
                                )
                                is not None
                                else check
                            )
                            for check in checks
                        ]

                        if verifier_diverged:
                            # The verifier's own recomputation blew up. The fault is
                            # in the verifier, so the next attempt must regenerate
                            # the VERIFIER rather than rewrite a solver whose numbers
                            # are plausible.
                            #
                            # Its own checks must not keep blocking either: a check
                            # that quotes a diverged number as its counterexample is
                            # not evidence about the solver. Downgrade exactly those
                            # checks to NOT_RUN, which still blocks the report (so no
                            # standard is relaxed) but states the honest reason — the
                            # value could not be verified — instead of asserting the
                            # solver is wrong. Measured on the real A problem, a
                            # verifier reported "max |dT| = 2.67...e+155 C" as its
                            # counterexample and that FAIL was the run's blocker.
                            checks = [
                                (
                                    check.model_copy(
                                        update={
                                            "status": CheckStatus.NOT_RUN,
                                            "detail": (
                                                "Not verifiable: this check rested on "
                                                f"the implausible value {implausible:.6g}, "
                                                "which comes from the verifier's own "
                                                "diverged recomputation rather than from "
                                                f"the solver. Original finding: {check.detail}"
                                            )[:2000],
                                        }
                                    )
                                    if (
                                        check.category == "independent"
                                        and check.status == CheckStatus.FAIL
                                        and (
                                            implausible := unevidenced_by_diverged_recomputation(
                                                check
                                            )
                                        )
                                        is not None
                                    )
                                    else check
                                )
                                for check in checks
                            ]
                            carried_verifier_feedback = [
                                "Your previous verifier program ran to completion but "
                                "its own recomputed values diverged: "
                                + ", ".join(sorted(verifier_diverged)[:8])
                                + ". Those values cannot be used as a reference, and "
                                "comparing them with the solver's finite, plausible "
                                "values produces a disagreement that says nothing "
                                "about the solver.",
                                "Recompute them with a numerically stable scheme: "
                                "prefer an implicit method over an explicit one on a "
                                "fine grid, keep the time step inside the stability "
                                "limit, and print the recomputed value alongside a "
                                "physical bound so a diverged number is visible. If a "
                                "quantity truly cannot be recomputed stably, omit that "
                                "key instead of reporting a diverged value.",
                            ]
                            logger.warning(
                                "Independent verifier recomputation diverged on %s",
                                ", ".join(sorted(verifier_diverged)[:8]),
                            )
                        if not verifier_summary.get("statistics"):
                            carried_verifier_feedback = [
                                "Your previous verifier program ran to completion but "
                                "printed no 'statistics' object, so its recomputed "
                                "values could not be compared with the claimed ones.",
                                "The final JSON line MUST include a 'statistics' object "
                                "holding the numbers you recomputed yourself, keyed by "
                                "the same names as the CLAIMED STATISTICS.",
                            ]
                        elif agreement.status == CheckStatus.FAIL:
                            # A disagreement is not proof that the solver is wrong:
                            # the verifier's own derivation may be the incorrect one,
                            # and blindly "repairing" a correct solver destroys a good
                            # solution. So the next attempt regenerates the verifier as
                            # well, with the disagreement spelled out, and asks it to
                            # justify its rule before accusing the solver.
                            carried_verifier_feedback = [
                                "Your previous verifier program recomputed values that "
                                "DISAGREE with the solver's claims:",
                                agreement.detail[:1200],
                                "Before concluding that the solver is wrong, re-derive "
                                "the rule you used to expand the input data into the "
                                "compared quantity, starting from the problem statement. "
                                "If the problem statement contains a worked example, a "
                                "sample computation, or an appendix with explicit values, "
                                "check your rule against it and report the comparison in "
                                "`evidence`. A verifier whose own rule contradicts the "
                                "statement's example is the thing that needs fixing.",
                            ]
                except Exception as exc:
                    checks.append(VerificationCheck(
                        name="independent_verifier_executed",
                        category="independent",
                        status=CheckStatus.FAIL,
                        detail=f"Independent verification could not be performed: {exc}",
                    ))
            else:
                checks.append(VerificationCheck(
                    name="independent_verifier_executed",
                    category="independent",
                    status=CheckStatus.NOT_RUN,
                    detail=(
                        "Skipped because the required output files are missing or "
                        "unreadable, so there is nothing to verify"
                    ),
                ))

            report = build_report(model, outcome.run_id, checks)
            _write_json(
                artifacts.verify / f"report{attempt}.json",
                report.model_dump(mode="json"),
            )

            if report.passed:
                state.complete(
                    "verify",
                    f"PASS — {len(checks)} check(s), 0 failures",
                    [str(artifacts.verify / f"report{attempt}.json")],
                )
                break

            feedback = report.to_prompt_feedback()
            # A repair that makes the scheme worse must be told so. Per-attempt
            # feedback says what is wrong NOW; it cannot say that the previous
            # attempt was closer, so the solver has no way to notice it is
            # oscillating or drifting away from a discretisation that nearly
            # worked. Measured on this project: the same problem yielded
            # err=0.475 on one run and err=57.319 on another, and repairs that
            # rewrote the scheme wholesale produced the worse one.
            self_check = (outcome.summary or {}).get("self_check") or {}
            code = getattr(program, "code", None)
            if code:
                attempt_programs[attempt] = code
            try:
                raw_error = float(self_check.get("max_abs_error"))
                raw_tolerance = float(self_check.get("tolerance"))
            except (TypeError, ValueError):
                raw_error = None
                raw_tolerance = None
            if (
                raw_error is not None
                and raw_tolerance is not None
                and raw_tolerance > 0.0
            ):
                attempt_error_ratios.append((attempt, raw_error / raw_tolerance))
                named = self_check.get("failed_criteria")
                if isinstance(named, str):
                    named = [named]
                if isinstance(named, list):
                    attempt_failed_criteria[attempt] = [
                        str(item) for item in named if str(item).strip()
                    ]
                trend = _worsening_trend_note(
                    attempt_error_ratios, attempt, attempt_failed_criteria
                )
                if trend:
                    feedback.insert(0, trend)
            # A bare numeric disagreement invites guesswork, so name the cause that
            # fits THIS model. Telling a PDE solver to re-derive a combinatorial
            # occupancy expansion sends it after a structure its problem does not
            # have, and it can never converge from there.
            if _independent_disagreement(report):
                feedback.append(_disagreement_hint(model))
            state.fail(
                "verify",
                f"attempt {attempt} FAILED: {len(report.blocking_failures)} issue(s)",
            )
            if attempt >= self._max_repair:
                state.block(
                    "verify",
                    "verification still failing after "
                    f"{self._max_repair} repair attempt(s)",
                )
                return self._result(
                    state,
                    blockers=report.blocking_failures[:10],
                )

            # A semantic disagreement with an independent recomputation usually
            # means the MODEL is wrong, not the code. Regenerating code from an
            # unchanged model can never converge, so repair the model first and
            # let the next attempt generate code from the corrected one.
            if _needs_model_repair(report):
                repaired = await self._repair_model(
                    state=state,
                    state_obj=state_obj,
                    artifacts=artifacts,
                    problem_context=problem_context,
                    feedback=report.blocking_failures[:8] + feedback[-2:],
                )
                if repaired is not None:
                    model = repaired
                    feedback = []

        if report is None or not report.passed or outcome is None or program is None:
            if report is not None:
                blockers = report.blocking_failures
            else:
                # No verification report exists, so report why: the loop ended
                # because generation or execution never produced a result.
                blockers = feedback or ["verification did not run"]
            return self._result(state, blockers=blockers[:10])

        # ── 11. Evidence package ──────────────────────────────
        state.begin("evidence", "registering verified evidence and claims")
        statistics = (outcome.summary or {}).get("statistics", {}) or {}
        evidence, claims = self._build_evidence(
            analysis=analysis,
            model=model,
            outcome=outcome,
            report=report,
            statistics=statistics,
        )
        _write_json(artifacts.evidence / "evidence.json", {
            "evidence": [e.model_dump(mode="json") for e in evidence.list_evidence()],
            "claims": [c.model_dump(mode="json") for c in evidence.list_claims()],
        })
        state.complete(
            "evidence",
            f"{len(evidence.list_evidence())} evidence ref(s), "
            f"{len(evidence.list_claims())} claim(s)",
        )

        # ── 12-15. Publishing the verified solution ───────────
        # Resuming must not re-draft a paper that is already written: the
        # figure proposal and the section drafts are model calls, and
        # regenerating them would replace a delivered paper with differently
        # worded prose about the same numbers. The publication is reused only
        # when the verified inputs behind it are unchanged.
        output_summaries = self._output_summaries(
            solve_dir, problem_context.required_outputs
        )
        publication_inputs = {
            "outcome": outcome.run_id,
            "model_version": model.version,
            "statistics": statistics,
            "subproblems": [s.model_dump(mode="json") for s in analysis.subproblems],
            "output_summaries": output_summaries,
        }
        reusable = self._reusable_publication(artifacts, state, publication_inputs)
        if reusable is not None:
            figures, tables, paper = reusable
            state.complete("figures", "reused from the previous run")
            state.complete("tables", "reused from the previous run")
            state.complete(
                "paper",
                f"{len(paper.sections)} section(s) reused from the previous run",
                [str(artifacts.paper / "paper_ir.json")],
            )
        else:
            # ── 12. Figures ───────────────────────────────────────
            state.begin("figures", "rendering figures from verified results")
            figure_builder = FigureBuilder(self._router)
            specs = await figure_builder.propose(
                statistics=statistics,
                output_summaries=output_summaries,
                subproblems=[s.model_dump(mode="json") for s in analysis.subproblems],
            )
            figures = figure_builder.render(
                specs=specs,
                statistics=statistics,
                output_dir=solve_dir,
                figures_dir=artifacts.figures,
                execution_id=outcome.run_id,
                source_data_ids=["EVD-EXECUTION"],
            )
            _write_json(
                artifacts.figures / "figures.json",
                [f.model_dump(mode="json") for f in figures.all()],
            )
            state.complete("figures", f"{len(figures.all())} figure(s) rendered")

            # ── 13. Tables ────────────────────────────────────────
            state.begin("tables", "building tables from verified results")
            table_builder = TableBuilder()
            tables = TableRegistry()
            stats_table = table_builder.build_statistics_table(statistics, "EVD-EXECUTION")
            if stats_table:
                tables.register(stats_table)
            for index, key in enumerate(_nested_stat_keys(statistics), start=2):
                table = table_builder.build_nested_statistics_table(
                    statistics=statistics,
                    source_id="EVD-EXECUTION",
                    table_id=f"TAB-{index:03d}",
                    title=f"求解结果统计：{key}",
                    key=key,
                )
                if table:
                    tables.register(table)
            _write_json(
                artifacts.tables / "tables.json",
                [t.model_dump(mode="json") for t in tables.all()],
            )
            state.complete("tables", f"{len(tables.all())} table(s) built")

            # ── 14. Paper context package ─────────────────────────
            package = self._context_package(
                context=problem_context,
                analysis=analysis,
                assumption_ledger=assumption_ledger,
                ambiguity_register=ambiguity_register,
                model=model,
                outcome=outcome,
                report=report,
                statistics=statistics,
                figures=figures,
                tables=tables,
                problem_context=problem_context,
                output_files=problem_context.required_outputs,
            )
            _write_json(artifacts.paper / "context_package.json", package.model_dump(mode="json"))

            # ── 15. Progressive paper writing ─────────────────────
            state.begin("paper", "outline → section drafting → assembly")
            writer = PaperWriter(self._router)
            outline = await self._retry_call(
                "paper-outline", lambda: writer.build_outline(package)
            )
            _write_json(artifacts.paper / "outline.json", outline.model_dump(mode="json"))

            drafts = []
            for index, section in enumerate(outline.sections, start=1):
                draft = await self._retry_call(
                    f"paper-section:{section.title}",
                    lambda s=section, i=index: writer.write_section(s, package, i),
                )
                drafts.append(draft)
                _write_json(
                    artifacts.paper / "sections" / f"{index:02d}_{section.title}.json",
                    draft.model_dump(mode="json"),
                )
            blocked_sections = [d.title for d in drafts if d.blocked]
            if blocked_sections:
                state.notes.append(
                    "Sections reported as blocked by the writer: " + "、".join(blocked_sections)
                )

            # A number in the prose that no verified source supports is a fabricated
            # figure. Rewrite only the sections that contain one, so the rest of the
            # paper is not disturbed.
            allowed_numbers = _traceable_numbers(
                statistics,
                tables,
                problem_context.problem_text,
                extra=self._verifiable_extras(report, output_summaries),
            )
            for _ in range(2):
                prose = " ".join(
                    paragraph for draft in drafts for paragraph in draft.paragraphs
                )
                unsupported = _untraceable_numbers(prose, allowed_numbers)
                if not unsupported:
                    break
                offenders = [
                    index for index, draft in enumerate(drafts)
                    if _untraceable_numbers(" ".join(draft.paragraphs), allowed_numbers)
                ]
                if not offenders:
                    break
                logger.warning(
                    "Rewriting %d section(s) that state unsupported numbers: %s",
                    len(offenders), sorted(unsupported),
                )
                correction = (
                    "Your previous draft stated numbers that no verified source "
                    "supports: " + "、".join(sorted(unsupported)) + ". "
                    "Remove or correct them. Every number you state must appear "
                    "verbatim in `verified_statistics`, in the output file summaries, "
                    "or in the problem statement. If you do not have a verified value, "
                    "do not state one."
                )
                for index in offenders:
                    drafts[index] = await self._retry_call(
                        f"paper-repair:{outline.sections[index].title}",
                        lambda i=index: writer.write_section(
                            outline.sections[i], package, i + 1, correction=correction
                        ),
                    )
                    _write_json(
                        artifacts.paper / "sections"
                        / f"{index + 1:02d}_{outline.sections[index].title}.json",
                        drafts[index].model_dump(mode="json"),
                    )

            abstract = await self._retry_call(
                "paper-abstract", lambda: writer.write_abstract(outline, package, drafts)
            )
            # The abstract is prose too, and it is the part every judge reads first.
            for _ in range(2):
                unsupported = _untraceable_numbers(abstract, allowed_numbers)
                if not unsupported:
                    break
                logger.warning(
                    "Rewriting the abstract; it states unsupported numbers: %s",
                    sorted(unsupported),
                )
                abstract = await self._retry_call(
                    "paper-abstract-repair",
                    lambda: writer.write_abstract(
                        outline, package, drafts,
                        correction=(
                            "Your previous abstract stated numbers that no verified source "
                            "supports: " + "、".join(sorted(unsupported)) + ". "
                            "Every number must appear verbatim in `verified_statistics` or in "
                            "the problem statement. Remove the rest."
                        ),
                    ),
                )
            remaining = _untraceable_numbers(abstract, allowed_numbers)
            if remaining:
                state.notes.append(
                    "abstract still states unsupported numbers after repair: "
                    + "、".join(sorted(remaining))
                )

            paper = drafts_to_paper_ir(
                title=outline.title or "数学建模竞赛论文",
                abstract=abstract,
                keywords=outline.keywords or ["数学建模", "优化", "算法"],
                drafts=drafts,
            )
            _ensure_results_visible(paper, figures, tables)
            _attach_claims(paper, evidence, statistics)
            _write_json(artifacts.paper / "paper_ir.json", paper.model_dump(mode="json"))
            state.complete(
                "paper",
                f"{len(paper.sections)} section(s), abstract {len(abstract)} chars",
                [str(artifacts.paper / "paper_ir.json")],
            )
            _write_json(
                artifacts.paper / "publication_inputs.json", publication_inputs
            )

        # ── 16. Submission audit ──────────────────────────────
        state.begin("audit", "SubmissionCheck")
        profile = CompetitionProfile(
            competition_name="CUMCM",
            page_limit=None,
            required_sections=[s.title for s in paper.sections],
            reference_style="bibtex",
        )
        checker = SubmissionCheckAgent(
            profile=profile,
            evidence=evidence,
            figures=figures,
            tables=tables,
            current_model_version=model.version,
            source_values={k: v for k, v in statistics.items() if isinstance(v, (int, float))},
        )
        submission = checker.check(paper)
        _write_json(artifacts.audit / "submission_check.json", submission.model_dump(mode="json"))
        state.complete(
            "audit",
            f"{submission.status.value}: {len(submission.failures)} failure(s), "
            f"{len(submission.warnings)} warning(s)",
        )

        # ── 17. PDF ───────────────────────────────────────────
        state.begin("pdf", "rendering LaTeX and compiling with xelatex")
        tex = build_cumcm_latex(paper, figures, tables, artifacts.paper)
        tex_path = artifacts.paper / "paper.tex"
        tex_path.write_text(tex, encoding="utf-8")
        pdf_result = build_pdf(tex, artifacts.paper, image=self._paper_image)
        _write_json(artifacts.paper / "pdf_build.json", pdf_result.model_dump(mode="json"))
        if not pdf_result.success:
            state.fail("pdf", pdf_result.reason or "PDF build failed")
            return self._result(state, blockers=[f"PDF build failed: {pdf_result.reason}"])
        state.complete("pdf", f"{pdf_result.pdf_size} bytes at {pdf_result.pdf_path}")

        # ── 18. Support package ───────────────────────────────
        state.begin("package", "assembling output/ and manifest.json")
        output_dir = run_dir / "output"
        builder = SupportPackageBuilder(output_dir)
        source_code = sorted(artifacts.code.rglob("solver.py")) + sorted(
            artifacts.verify.rglob("verifier.py")
        )
        processed = [
            Path(p) for p in (self._load_json(artifacts.intake / "processed_data.json") or [])
        ]
        originals = [Path(p) for p in state.input_files if Path(p).exists()]
        result_files = [
            solve_dir / name
            for name in problem_context.required_outputs
            if (solve_dir / name).exists()
        ]
        table_files = _export_table_files(tables, artifacts.tables / "files")
        manifest = builder.build(
            paper_pdf=Path(pdf_result.pdf_path),
            paper_tex=tex_path,
            source_code=source_code,
            processed_data=processed,
            original_files=originals,
            figures=[Path(f.artifact_path) for f in figures.all()],
            tables=table_files,
            result_files=result_files,
            manifest_extra={
                "run_id": state.run_id,
                "competition": competition_name(problem_context),
                "model": {"model_id": model.model_id, "name": model.name, "version": model.version},
                "verification": {
                    "overall": report.overall.value,
                    "checks": [
                        {"name": c.name, "category": c.category, "status": c.status.value}
                        for c in report.checks
                    ],
                },
                "submission_check": {
                    "status": submission.status.value,
                    "failures": submission.failures,
                },
                "data_status": DATA_STATUS_FORMAL,
                "statistics": statistics,
                "result_files": [p.name for p in result_files],
                "tables": [
                    {"table_id": t.table_id, "title": t.title, "rows": len(t.rows)}
                    for t in tables.all()
                ],
                "figures": [
                    {"figure_id": f.figure_id, "title": f.title} for f in figures.all()
                ],
            },
        )
        state.complete("package", f"{len(manifest['contents'])} entries packaged")

        # ── 19. Final consistency check ───────────────────────
        state.begin("final_check", "verifying deliverable consistency")
        final_issues, final_warnings = self._final_check(
            output_dir=output_dir,
            paper=paper,
            figures=figures,
            tables=tables,
            statistics=statistics,
            result_files=result_files,
            pdf_result=pdf_result,
            problem_context_text=problem_context.problem_text,
            report=report,
            output_summaries=output_summaries,
        )
        _write_json(
            artifacts.audit / "final_check.json",
            {"issues": final_issues, "warnings": final_warnings},
        )
        if final_warnings:
            state.notes.extend(final_warnings)
        if final_issues:
            state.fail("final_check", "; ".join(final_issues)[:800])
            return self._result(state, blockers=final_issues)
        state.complete(
            "final_check",
            "all consistency checks passed"
            + (f" ({len(final_warnings)} warning(s))" if final_warnings else ""),
        )

        state.status = RunStatus.COMPLETED
        state.save()
        return self._result(state)

    # ══════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════

    def _result(
        self,
        state: RunState,
        blockers: Optional[list[str]] = None,
    ) -> AutopilotResult:
        run_dir = Path(state.run_dir)
        output_dir = run_dir / "output"
        pdf_path = output_dir / "paper.pdf"

        # A run that returns with blockers has stopped, whatever stage recorded
        # the failure: `state.fail()` only marks the stage, so a solve failure
        # followed by a failed verification could leave the run announcing
        # `running` forever. A caller cannot tell a finished-but-blocked run from
        # one that is still working, so settle the status here.
        if blockers and state.status == RunStatus.RUNNING:
            state.status = RunStatus.BLOCKED
            state.save()

        verify_path = None
        reports = list((run_dir / "artifacts" / "verify").glob("report*.json"))
        if reports:
            # Report the verdict that gated THIS run. Picking the highest-numbered
            # file reports a stale failure from an earlier attempt, so a run that
            # actually passed still announces "FAIL".
            verify_path = max(reports, key=lambda p: p.stat().st_mtime)
            verification = json.loads(verify_path.read_text(encoding="utf-8")).get("overall", "")
        else:
            verification = ""

        # Persist notes and status gathered after the last stage change (final
        # warnings, ledger failures) and hand the finished run to the ledger.
        state.save()

        return AutopilotResult(
            run_id=state.run_id,
            status=state.status.value,
            run_dir=str(run_dir),
            output_dir=str(output_dir) if output_dir.exists() else "",
            paper_pdf=str(pdf_path) if pdf_path.exists() else "",
            verification=verification,
            pending_questions=state.pending_questions,
            stage_summary=state.summary()["stages"],
            blockers=list(blockers or []),
            notes=list(state.notes),
        )

    @staticmethod
    def _agent_problem_text(context: ProblemContext) -> str:
        """Problem statement plus the real, parsed attachment schema.

        Agents reason over what was actually read from the files, not over
        filenames or assumptions about what the attachments contain.
        """
        parts = [context.problem_text.strip()]
        if context.attachments:
            parts.append("")
            parts.append(context.to_prompt(max_sheet_rows=3))
        if context.required_outputs:
            parts.append("")
            parts.append(
                "REQUIRED OUTPUT FILES: " + ", ".join(context.required_outputs)
            )
        return "\n".join(parts)[:40000]

    @staticmethod
    def _load_json(path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    async def _attempt(self, stage: str, call, max_attempts: int = MAX_AGENT_ATTEMPTS):
        """Run an LLM-backed stage, retrying transient failures.

        Structured generation can fail on a truncated or invalid body; that is
        a per-call accident, not a property of the problem, so retry a bounded
        number of times before declaring the stage failed.
        """
        last = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = await call()
            except Exception as exc:
                logger.warning("%s attempt %d raised: %s", stage, attempt, exc)
                last = None
                continue
            if result is not None and getattr(result, "status", None) == AgentStatus.COMPLETED:
                return result
            last = result
            detail = ""
            if result is not None:
                detail = "; ".join(e.message for e in result.errors)[:200]
            logger.warning(
                "%s attempt %d did not complete: %s", stage, attempt, detail or "unknown"
            )
        return last

    def _sandbox_inputs(self, artifacts: "_ArtifactPaths", context: ProblemContext) -> dict[str, str]:
        """Every real input the solver program is allowed to read."""
        inputs: dict[str, str] = {}
        for path in sorted(artifacts.processed_data.glob("*")):
            if path.is_file():
                inputs[path.name] = path.read_text(encoding="utf-8", errors="replace")
        return inputs

    def _template_headers(
        self,
        artifacts: "_ArtifactPaths",
        context: ProblemContext,
    ) -> dict[str, list[str]]:
        from mathmodel.autopilot.verify import read_output_table

        headers: dict[str, list[str]] = {}
        for info in context.attachments:
            if info.role != "template":
                continue
            try:
                parsed_headers, _ = read_output_table(Path(info.storage_path))
            except Exception:
                continue
            headers[info.original_name] = parsed_headers
        return headers

    def _template_sheets(
        self,
        context: ProblemContext,
    ) -> dict[str, list[str]]:
        """Worksheet names each template file declares."""
        from mathmodel.autopilot.verify import read_output_sheet_names

        sheets: dict[str, list[str]] = {}
        for info in context.attachments:
            if info.role != "template":
                continue
            try:
                sheets[info.original_name] = read_output_sheet_names(
                    Path(info.storage_path)
                )
            except Exception:
                continue
        return sheets

    @staticmethod
    def _expected_row_counts(
        context: ProblemContext,
        template_headers: dict[str, list[str]],
    ) -> dict[str, int]:
        """Row counts a per-plan template has to be filled to.

        A template whose first column is 用频装备编号 is a per-plan table, so the
        answer must state the disposition of every plan in the data, not only the
        ones that were changed.
        """
        plan_count = max(
            (
                sheet.row_count
                for info in context.attachments
                if info.role == "data"
                for sheet in info.sheets
            ),
            default=0,
        )
        if plan_count <= 0:
            return {}
        return {
            name: plan_count
            for name, headers in template_headers.items()
            if headers and "编号" in headers[0]
        }

    def _output_summaries(
        self,
        solve_dir: Path,
        required_outputs: list[str],
    ) -> dict[str, dict[str, Any]]:
        from mathmodel.autopilot.verify import read_output_table

        summaries: dict[str, dict[str, Any]] = {}
        for name in required_outputs:
            path = solve_dir / name
            if not path.exists():
                continue
            try:
                headers, rows = read_output_table(path)
            except Exception:
                continue
            summaries[name] = {
                "headers": headers,
                "row_count": len(rows),
                "sample_rows": [[("" if c is None else c) for c in r] for r in rows[:5]],
            }
        return summaries

    # ── Registries ────────────────────────────────────────────

    @staticmethod
    def _ambiguity_register(analysis) -> dict[str, Any]:
        return {
            "generated_from": analysis.analysis_id,
            "ambiguities": [
                {
                    "ambiguity_id": a.ambiguity_id,
                    "description": a.description,
                    "interpretations": a.interpretations,
                    "preferred_interpretation": a.preferred_interpretation,
                    "preference_reason": a.preference_reason,
                    "impact": a.impact.value,
                    "requires_human_review": a.requires_human_review,
                }
                for a in analysis.ambiguities
            ],
        }

    @staticmethod
    def _assumption_ledger(analysis) -> dict[str, Any]:
        assumptions = []
        for item in analysis.evidence:
            if getattr(item, "is_assumption", False):
                assumptions.append({
                    "assumption_id": item.evidence_id,
                    "text": item.content,
                    "source": item.source,
                    "status": "accepted",
                })
        for condition in analysis.implicit_conditions:
            assumptions.append({
                "assumption_id": f"IMP-{len(assumptions) + 1}",
                "text": condition,
                "source": "implicit_condition",
                "status": "accepted",
            })
        return {"assumptions": assumptions}

    @staticmethod
    def _data_schema(context: ProblemContext) -> dict[str, Any]:
        tables = []
        for info, sheet in context.data_sheets():
            tables.append({
                "file": info.original_name,
                "sheet": sheet.name,
                "row_count": sheet.row_count,
                "headers": sheet.headers,
                "columns": [
                    {
                        "name": c.name,
                        "dtype": c.dtype,
                        "semantic_type": c.semantic_type.value,
                        "missing_count": c.missing_count,
                        "unique_count": c.unique_count,
                        "example_values": [str(v) for v in c.example_values[:5]],
                    }
                    for c in sheet.columns
                ],
                "sample_rows": sheet.sample_rows,
            })
        return {"tables": tables}

    @staticmethod
    def _selection_audit(candidates: list[ModelCandidate], jury_result) -> dict[str, Any]:
        scores = []
        for score in (getattr(jury_result, "candidate_scores", None) or []):
            scores.append({
                "candidate_id": getattr(score, "candidate_id", ""),
                "total_score": getattr(score, "total_score", None),
                "dimension_scores": getattr(score, "dimension_scores", None),
                "rationale": getattr(score, "rationale", ""),
            })
        return {
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "name": c.name,
                    "model_family": getattr(c.model_family, "value", str(c.model_family)),
                    "strengths": c.strengths,
                    "weaknesses": c.weaknesses,
                }
                for c in candidates
            ],
            "selected": jury_result.selected_model,
            "backup": jury_result.backup_model,
            "ranking": list(getattr(jury_result, "ranking", []) or []),
            "rejected": list(getattr(jury_result, "rejected_models", []) or []),
            "scores": scores,
            "rationale": getattr(jury_result, "decision_reason", "") or "",
            "selection_rule": "deterministic weighted scoring (ModelJury)",
        }

    # ── Evidence ──────────────────────────────────────────────

    @staticmethod
    def _build_evidence(
        analysis,
        model: MathematicalModel,
        outcome,
        report,
        statistics: dict[str, Any],
    ) -> tuple[EvidenceStore, list[Claim]]:
        store = EvidenceStore()

        store.register_evidence(EvidenceRef(
            evidence_id="EVD-PROBLEM",
            source_type=EvidenceSourceType.PROBLEM_FACT,
            source_id=analysis.analysis_id,
            description="Parsed competition problem statement",
        ))
        store.register_evidence(EvidenceRef(
            evidence_id="EVD-MODEL",
            source_type=EvidenceSourceType.MATHEMATICAL_MODEL,
            source_id=model.model_id,
            description=model.name,
            metadata={"model_version": model.version},
        ))
        store.register_evidence(EvidenceRef(
            evidence_id="EVD-EXECUTION",
            source_type=EvidenceSourceType.EXECUTION,
            source_id=outcome.run_id,
            description="Real sandbox execution of the generated solver program",
            metadata={"model_version": model.version, "image": outcome.sandbox.get("image")},
        ))
        store.register_evidence(EvidenceRef(
            evidence_id="EVD-SOLVER-RESULT",
            source_type=EvidenceSourceType.SOLVER_RESULT,
            source_id=outcome.run_id,
            description="Computed statistics returned by the executed program",
            metadata={"model_version": model.version},
        ))
        store.register_evidence(EvidenceRef(
            evidence_id="EVD-VALIDATION",
            source_type=EvidenceSourceType.VALIDATION,
            source_id=report.report_id,
            description=f"Verification gate result: {report.overall.value}",
            metadata={"model_version": model.version},
        ))

        claims: list[Claim] = []
        for index, (key, value) in enumerate(statistics.items(), start=1):
            if not isinstance(value, (int, float, str)):
                continue
            claim = Claim(
                claim_id=f"CLAIM-STAT-{index:03d}",
                text=f"{key} = {value}",
                claim_type=ClaimType.NUMERICAL,
                importance="high",
                evidence_ids=["EVD-SOLVER-RESULT", "EVD-EXECUTION"],
                verification_status=ClaimStatus.SUPPORTED,
                created_by="Autopilot",
                metadata={"statistic_key": key, "statistic_value": value},
            )
            store.register_claim(claim)
            claims.append(claim)

        model_claim = Claim(
            claim_id="CLAIM-MODEL",
            text=model.description or model.name,
            claim_type=ClaimType.MATHEMATICAL,
            importance="high",
            evidence_ids=["EVD-MODEL"],
            verification_status=ClaimStatus.SUPPORTED,
            created_by="Autopilot",
        )
        store.register_claim(model_claim)
        claims.append(model_claim)

        for index, equation in enumerate(model.equations, start=1):
            store.register_evidence(EvidenceRef(
                evidence_id=f"EVD-EQ-{index:03d}",
                source_type=EvidenceSourceType.EQUATION,
                source_id=equation.equation_id,
                description=equation.name,
            ))
        if model.equations:
            eq_claim = Claim(
                claim_id="CLAIM-EQUATIONS",
                text="模型方程见模型建立部分",
                claim_type=ClaimType.MATHEMATICAL,
                importance="normal",
                evidence_ids=[f"EVD-EQ-{i:03d}" for i in range(1, len(model.equations) + 1)],
                verification_status=ClaimStatus.SUPPORTED,
                created_by="Autopilot",
            )
            store.register_claim(eq_claim)
            claims.append(eq_claim)

        conclusion = Claim(
            claim_id="CLAIM-CONCLUSION",
            text=f"模型通过验证：{report.overall.value}",
            claim_type=ClaimType.CONCLUSION,
            importance="high",
            evidence_ids=["EVD-VALIDATION", "EVD-SOLVER-RESULT"],
            verification_status=ClaimStatus.SUPPORTED,
            created_by="Autopilot",
        )
        store.register_claim(conclusion)
        claims.append(conclusion)

        return store, claims

    @staticmethod
    def _context_package(
        context: ProblemContext,
        analysis,
        assumption_ledger: dict[str, Any],
        ambiguity_register: dict[str, Any],
        model: MathematicalModel,
        outcome,
        report,
        statistics: dict[str, Any],
        figures: FigureRegistry,
        tables: TableRegistry,
        problem_context: ProblemContext,
        output_files: list[str],
    ) -> PaperContextPackage:
        return PaperContextPackage(
            problem_title=model.name,
            problem_text=context.problem_text,
            subproblems=[s.model_dump(mode="json") for s in analysis.subproblems],
            objectives=analysis.objectives,
            constraints=analysis.explicit_constraints + analysis.implicit_conditions,
            ambiguities=ambiguity_register["ambiguities"],
            assumptions=assumption_ledger["assumptions"],
            model_summary={
                "model_id": model.model_id,
                "name": model.name,
                "description": model.description,
                "model_family": model.model_family,
                "algorithm_plan": model.algorithm_plan,
                "variables": [
                    {"symbol": v.symbol, "name": v.name, "meaning": v.meaning, "unit": v.unit}
                    for v in model.variables
                ],
                "parameters": [
                    {"symbol": p.symbol, "name": p.name, "value": p.value, "unit": p.unit}
                    for p in model.parameters
                ],
                "constraints": [
                    {"name": c.name, "expression": c.expression, "relation": c.relation.value, "rhs": c.rhs}
                    for c in model.constraints
                ],
            },
            equations=[
                {"equation_id": e.equation_id, "name": e.name, "latex": e.latex or e.expression}
                for e in model.equations
            ],
            solver_approach=outcome.summary.get("notes", "") if outcome.summary else "",
            verified_statistics=statistics,
            verification_summary={
                "overall": report.overall.value,
                "checks": [
                    {"name": c.name, "category": c.category, "status": c.status.value, "detail": c.detail}
                    for c in report.checks
                ],
            },
            output_files=output_files,
            figures=[
                {
                    "figure_id": f.figure_id,
                    "title": f.title,
                    "caption": f.caption,
                    "source": f.metadata.get("source"),
                }
                for f in figures.all()
            ],
            tables=[
                {
                    "table_id": t.table_id,
                    "title": t.title,
                    "headers": t.headers,
                    "row_count": len(t.rows),
                }
                for t in tables.all()
            ],
            references=[],
            limitations=[],
            data_status=DATA_STATUS_FORMAL,
        )

    # ── Final check ───────────────────────────────────────────

    @staticmethod
    def _final_check(
        output_dir: Path,
        paper,
        figures: FigureRegistry,
        tables: TableRegistry,
        statistics: dict[str, Any],
        result_files: list[Path],
        pdf_result,
        problem_context_text: str = "",
        report=None,
        output_summaries: Optional[dict[str, Any]] = None,
    ) -> tuple[list[str], list[str]]:
        """Hard requirements that must hold, plus reportable consistency warnings."""
        issues: list[str] = []
        warnings: list[str] = []

        pdf_path = output_dir / "paper.pdf"
        if not pdf_path.exists():
            issues.append("output/paper.pdf is missing")
        else:
            if pdf_path.read_bytes()[:5] != b"%PDF-":
                issues.append("output/paper.pdf is not a valid PDF file")
            if pdf_path.stat().st_size < 5000:
                issues.append("output/paper.pdf is suspiciously small")

        if not (output_dir / "manifest.json").exists():
            issues.append("output/manifest.json is missing")

        for figure in figures.all():
            if not Path(figure.artifact_path).exists():
                issues.append(f"figure artifact missing: {figure.artifact_path}")

        if not (output_dir / "support" / "necessary_supporting_files" / "paper.tex").exists():
            issues.append("paper.tex missing from support material")

        if not result_files:
            issues.append("no result files were produced")

        # The paper must actually display its verified results.
        if not paper.collect_tables():
            issues.append("the paper references no result table")
        if not paper.collect_figures():
            issues.append("the paper references no figure")

        paper_text = " ".join(
            block.text for section in paper.sections for block in section.content_blocks
        )

        # Every verified headline number must be reproducible from the paper's
        # own tables, so text, tables and the PDF cannot drift apart.
        table_values = {
            str(cell.value)
            for table in tables.all()
            for row in table.rows
            for cell in row
            if cell.value is not None
        }
        missing_numbers = []
        for key, value in statistics.items():
            if not isinstance(value, (int, float)) or abs(float(value)) < 1e-9:
                continue
            if _number_present(float(value), paper_text):
                continue
            if any(_number_present(float(value), cell) for cell in table_values):
                continue
            missing_numbers.append(f"{key}={value}")
        if missing_numbers:
            issues.append(
                "verified numbers absent from the paper: " + ", ".join(missing_numbers[:8])
            )

        placeholders = ["TODO", "PLACEHOLDER", "待补充", "XXX", "Lorem ipsum"]
        for token in placeholders:
            if token.lower() in paper_text.lower():
                issues.append(f"placeholder text left in the paper: {token}")

        # Numbers written in prose that cannot be traced to any verified source
        # are reported for review rather than silently accepted.
        untraceable = _untraceable_numbers(
            paper_text,
            _traceable_numbers(
                statistics,
                tables,
                problem_context_text,
                extra=(
                    _verification_numbers(report)
                    | (
                        _collect_into(output_summaries)
                        if output_summaries else set()
                    )
                ),
            ),
        )
        if untraceable:
            warnings.append(
                "numbers in the paper not traceable to verified results: "
                + ", ".join(sorted(untraceable)[:12])
            )

        return issues, warnings


class _ArtifactPaths:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.intake = run_dir / "artifacts" / "intake"
        self.analysis = run_dir / "artifacts" / "analysis"
        self.data = run_dir / "artifacts" / "data"
        self.models = run_dir / "artifacts" / "models"
        self.model = run_dir / "artifacts" / "model"
        self.code = run_dir / "artifacts" / "code"
        self.solve = run_dir / "artifacts" / "solve"
        self.verify = run_dir / "artifacts" / "verify"
        self.evidence = run_dir / "artifacts" / "evidence"
        self.figures = run_dir / "artifacts" / "figures"
        self.tables = run_dir / "artifacts" / "tables"
        self.paper = run_dir / "artifacts" / "paper"
        self.audit = run_dir / "artifacts" / "audit"
        self.processed_data = run_dir / "processed_data"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _json_key(payload: Any) -> str:
    """A stable string for comparing two JSON payloads.

    Values that do not survive a JSON round trip are compared as their
    `str()` form on both sides, so a reloaded payload still compares equal to
    the object it was written from.
    """
    return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)


def _nested_stat_keys(statistics: dict[str, Any]) -> list[str]:
    return [k for k, v in statistics.items() if isinstance(v, dict) and v]


# Markers of one specific, repeated failure on this project: the number of grid
# nodes and the length of a profile (or the number of output columns) disagreed.
# Six consecutive repair attempts on the real A problem each surfaced a different
# symptom of that single unresolved off-by-one -- `broadcast shapes (40,) (41,)`,
# `22 columns passed, passed data had 21 columns`, `header/data mismatch: hdr=22
# T=21 C=21`, `fp and xp are not of the same length`, `grid size 21 != profile
# length 20` -- and every repair patched the call site the error happened to
# surface at, so the error simply moved to the next one.
_SHAPE_MISMATCH_MARKERS = (
    "could not be broadcast together with shapes",
    "are not of the same length",
    "columns passed, passed data had",
    "header/data mismatch",
    "grid size",
    "profile length",
    # Measured on this project: a run failed four consecutive attempts with
    # "IndexError: invalid index to scalar variable" twice, then "ValueError:
    # could not broadcast input array from shape (2,) into shape (20,)", then
    # "TypeError: 'float' object is not subscriptable". All three are ONE bug --
    # a scalar where an array was expected -- but none of the markers above
    # matched, so the classifier saw three unrelated classes and the escalation
    # below never fired. The error moved; the bug did not.
    "invalid index to scalar variable",
    "object is not subscriptable",
    "could not broadcast input array",
    # Round 70 audit of 200 real solve outcomes: these are the SAME array/scalar
    # convention bug, and each was being classified as its own bare signature, so
    # the escalation never saw them repeat. Occurrence counts from that audit are
    # in the comments.
    "truth value of an array with more than one element is ambiguous",  # 6
    "setting an array element with a sequence",  # 5
    "only 0-dimensional arrays can be converted",  # 3
    "must have same number of dimensions",  # 1
    "too many indices for array",  # 1
    "len() of unsized object",  # 1
)

# Round 93: a recurring failure that no class grouped, so the escalation never
# fired. It is an output-bookkeeping error, not an array-shape error, so it gets
# its own class rather than sharing the shape note -- that note tells the repair
# the node count and the profile length disagree, which would misdirect it.
_OUTPUT_BOOKKEEPING_MARKERS = (
    "at least one sheet must be visible",  # 3
    "no worksheet named",
    "worksheet index",
)

_OUTPUT_BOOKKEEPING_NOTE = (
    "STOP PATCHING THE CALL SITE. This is attempt {attempt} and the previous "
    "{seen} attempt(s) failed while BUILDING THE OUTPUT WORKBOOK, not while "
    "solving. The solver ran; the error came from the file-writing step. Do not "
    "add another try/except around the write, and do not change the numerics. "
    "Instead make the workbook construction unconditional: create or select the "
    "target worksheet BEFORE writing any cell, never delete the last remaining "
    "worksheet, and write a row only after its sheet exists. Keep the sheet "
    "names the template requires -- renaming or removing a template sheet to "
    "work around this is not a fix."
)

_SHAPE_MISMATCH_NOTE = (
    "STOP PATCHING CALL SITES. This is attempt {attempt} and the previous {seen} "
    "attempt(s) failed the same class of error: the number of grid nodes and the "
    "length of a profile (or the number of output columns) disagree. Each earlier "
    "repair fixed the line where the error surfaced, which is why the error moved "
    "to a different line instead of going away. Do not add another assertion, "
    "reshape, or interpolation call. Instead find the single place that defines "
    "the node count and the single place that defines the output column count, "
    "make them agree, and derive every other length in the program from one "
    "constant. The template fixes the output columns; the internal node count is "
    "your own choice and must not determine the output shape."
)

_REPEATED_FAILURE_NOTE = (
    "STOP PATCHING CALL SITES. This is attempt {attempt} and the previous {seen} "
    "attempt(s) failed the same way ({cls}). A repair that makes the error move "
    "to a different line has not fixed anything: the fault is in a shared "
    "definition, not at the site that reported it. Locate the one place that "
    "defines the quantity both sides disagree about, change it there, and derive "
    "the rest from that single definition."
)

# The solver's own guard that the problem's final target was met. Measured on
# this project: attempts 2, 3, 4 and 6 all aborted on this assert at the
# IDENTICAL value 2.5217 (the moisture never dried), each time with the message
# reworded -- "final row", then "final written 60 s row", then "problem 3 final
# written 60 s row". The rewording is the tell: the repair was editing the
# guard's text instead of the physics, and a signature that strips only numbers
# saw three different failures instead of one repeated one.
_FINAL_TARGET_MARKERS = (
    # "strictly below" subsumes both "not strictly below" and the solver's own
    # wording "did not fall strictly below"; the narrower marker missed the
    # latter. Measured in the round-70 audit of 200 real outcomes.
    "strictly below",
    "not < 0.15",
    "is not below",
    "not strictly less than",
    "final row",
    "final written",
    # The solver's own guard is the most common phrasing in practice and was
    # matching NONE of the markers above, so these target failures were being
    # treated as unrelated one-off errors rather than one repeating class.
    "target not satisfied",  # 2
    "final max rounded",  # 3
    "strict criterion failed",  # 2
    "target not met",
)


def _repeated_final_target_note(attempt: int, seen: int) -> str:
    """Tell the repair to fix the transport, not the wording of its own guard."""
    return (
        f"YOUR OWN GUARD KEEPS TRIPPING, and you keep rewording it. This is "
        f"attempt {attempt} and the previous {seen} attempt(s) ended the same way: "
        f"your assertion that the final moisture is strictly below the required "
        f"threshold failed. Do NOT reword, relax or remove that assertion -- it is "
        f"correctly reporting that your drying is far too weak, and silencing it "
        f"would only hide the fault. The quantity it protects has not moved "
        f"between attempts, so the repair has been aimed at the message rather "
        f"than at the transport. Fix the mass transfer: check the surface "
        f"conductance is measured on the surface area, that the diffusion law's "
        f"exponent is the ratio the statement prints, and that the moisture "
        f"boundary condition actually acts on the surface node."
    )


def _failure_signature(text: str) -> str:
    """Collapse a failure message to a class key, ignoring the numbers in it."""
    stripped = (text or "").strip()
    if not stripped:
        return "unknown"
    line = stripped.splitlines()[-1]
    return re.sub(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?", "#", line)[:200].strip().lower()


def _failure_class(text: str) -> str:
    """Group a solver failure into a class the repair loop can notice repeating."""
    lowered = (text or "").lower()
    for marker in _OUTPUT_BOOKKEEPING_MARKERS:
        if marker in lowered:
            return "output-bookkeeping"
    for marker in _SHAPE_MISMATCH_MARKERS:
        if marker in lowered:
            return "shape-mismatch"
    # Checked before the numeric signature, so that a reworded but identical
    # failure stays ONE class instead of looking like several.
    for marker in _FINAL_TARGET_MARKERS:
        if marker in lowered:
            return "final-target-unmet"
    return _failure_signature(text)


def _worsening_trend_note(
    history: list[tuple[int, float]],
    attempt: int,
    criteria: Optional[dict[int, list[str]]] = None,
) -> Optional[str]:
    """Tell the repair when its own attempts are getting WORSE, not better.

    Per-attempt feedback describes the current failure. It cannot say that an
    earlier attempt was closer, so a solver rewriting wholesale has no way to
    notice it is drifting away from a discretisation that nearly worked.
    Measured on this project: one attempt reached err=0.475 against its own
    tolerance while a later one reached 57.319, on the same problem.

    When the failed-criteria names are available they are included, because
    "restore attempt 2" is only actionable if the solver can see WHICH criteria
    attempt 2 had already satisfied.
    """
    scored = [(number, ratio) for number, ratio in history if ratio > 0.0]
    if len(scored) < 2:
        return None
    best_attempt, best_ratio = min(scored, key=lambda item: item[1])
    current = next((ratio for number, ratio in scored if number == attempt), None)
    if current is None or current <= best_ratio:
        return None
    note = (
        f"Your repairs are making the scheme WORSE, not better. Attempt "
        f"{best_attempt} reached an error of {best_ratio:.4g} times its own "
        f"tolerance, while attempt {attempt} reaches {current:.4g} times. Do not "
        f"continue rewriting the scheme from the latest version: identify what "
        f"attempt {best_attempt} did differently and restore that, then change "
        f"only what the current issues require. A wholesale rewrite has already "
        f"been shown to move away from the better discretisation."
    )
    if criteria:
        better = criteria.get(best_attempt) or []
        worse = criteria.get(attempt) or []
        if better or worse:
            note += (
                f" Attempt {best_attempt} reported these criteria as failed: "
                f"{', '.join(better) if better else '(none)'}; attempt {attempt} "
                f"reports: {', '.join(worse) if worse else '(none)'}. Any "
                f"criterion that the better attempt had satisfied is something "
                f"your repair has broken."
            )
    return note


def _repair_base_program(
    history: list[tuple[int, float]],
    programs: dict[int, str],
    attempt: int,
    fallback: Optional[str],
) -> tuple[Optional[str], Optional[int]]:
    """The program a repair should be based on, and the attempt it came from.

    I43 tells the repair, in words, that it is drifting away from a better
    attempt and should restore it. That is a request. This makes it structural:
    when the current attempt is worse than the best one on record, the repair is
    handed the BEST attempt's program as its starting point rather than the
    worse program it has just written, so drifting away no longer depends on the
    model choosing to obey.

    Only used when at least two attempts carry a comparable error ratio and the
    current one is strictly worse than the best; otherwise the ordinary
    previous-attempt base is kept, since a monotone run has nothing to restore.
    """
    scored = [(number, ratio) for number, ratio in history if number in programs]
    if len(scored) < 2:
        return fallback, None
    best_attempt, best_ratio = min(scored, key=lambda pair: pair[1])
    current = dict(history).get(attempt)
    if current is None or best_attempt == attempt or current <= best_ratio:
        return fallback, None
    return programs[best_attempt], best_attempt


def _repeated_failure_note(failure_class: str, attempt: int, seen: int) -> str:
    """Escalate when a repair loop is repeating itself instead of converging."""
    if failure_class == "shape-mismatch":
        return _SHAPE_MISMATCH_NOTE.format(attempt=attempt, seen=seen)
    if failure_class == "output-bookkeeping":
        return _OUTPUT_BOOKKEEPING_NOTE.format(attempt=attempt, seen=seen)
    if failure_class == "final-target-unmet":
        return _repeated_final_target_note(attempt, seen)
    return _REPEATED_FAILURE_NOTE.format(
        attempt=attempt, seen=seen, cls=failure_class
    )


def _independent_disagreement(report) -> bool:
    """Whether an independent recomputation disagreed with the solver's numbers.

    This decides which cause-level hint the repair loop adds. The hint has to fit
    the model: a combinatorial explanation is useless, and actively misleading,
    to a solver whose problem integrates a differential equation.
    """
    for check in list(getattr(report, "checks", []) or []):
        if check.status != CheckStatus.FAIL or "independent" not in check.category:
            continue
        detail = check.detail or ""
        if any(
            word in detail
            for word in ("recomputed", "claimed", "solver=", "independent=")
        ):
            return True
    return False


def _disagreement_hint(model) -> str:
    """The cause-level hint to add when an independent recomputation disagrees.

    The hint must fit the model. A combinatorial explanation is useless, and
    actively misleading, to a solver whose problem integrates a differential
    equation: it sends the solver after a structure its problem does not have,
    and the repair loop cannot converge from there.
    """
    if model_integrates_differential_equations(model):
        return (
            "An independent recomputation of the same quantity from the raw input "
            "disagrees with your program. For a model that integrates a differential "
            "equation this normally means the NUMERICAL SCHEME is wrong, not that the "
            "arithmetic is. Check in this order: (1) every term of the control-volume "
            "balance must carry the same control-volume measure — an interior "
            "diffusion coefficient that dropped the cell width, or a storage term "
            "that mixes a per-node with a per-volume form, scales the entire scheme "
            "and makes it heat or freeze incorrectly; (2) any Picard / fixed-point "
            "loop must keep the PREVIOUS time level on the time-derivative right-hand "
            "side, because rebuilding that right-hand side from the current iterate "
            "converges to a different fixed point — raising the iteration count then "
            "makes the answer worse, not better; (3) the boundary conditions and the "
            "property closures must match the model's equations. Then validate the "
            "corrected scheme against a case with a known exact answer and report the "
            "measured deviation in `self_check` before re-solving."
        )
    return (
        "An independent recomputation of the same quantity from the raw input "
        "disagrees with your program. This normally means your occupancy expansion "
        "is wrong — re-read the problem statement's definition of a repeated use and "
        "any worked example it gives, re-derive the start of the k-th use, and assert "
        "your expansion reproduces that example before re-solving."
    )


def _needs_model_repair(report) -> bool:
    """Whether verification showed the mathematical model itself is wrong.

    Only a disagreement about a quantity that is a pure function of the INPUT
    implicates the model. Two things must hold for that to be meaningful:

    * the verifier actually read the input data, otherwise its "recomputation"
      is a no-op and the disagreement says nothing about the model; and
    * the disagreement is about an input-derived quantity. A solution that still
      violates constraints, or scores worse than a baseline, is a failure of the
      solving step — the model is fine and re-formalizing it discards good work.
    """
    checks = list(getattr(report, "checks", []) or [])
    if any(
        c.status == CheckStatus.FAIL and c.name == "independent_read_the_input"
        for c in checks
    ):
        return False
    for check in checks:
        if check.status != CheckStatus.FAIL or "independent" not in check.category:
            continue
        name = (check.name or "").lower()
        detail = check.detail or ""
        if "recomputation" in name or "recomputed_total" in detail:
            return True
        if "recomputed" in detail and "claimed" in detail:
            return True
    return False


def _collect_numbers(node: Any, into: set[float]) -> None:
    """Every number anywhere in a nested statistics payload."""
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        into.add(float(node))
        return
    if isinstance(node, dict):
        for value in node.values():
            _collect_numbers(value, into)
        return
    if isinstance(node, (list, tuple, set)):
        for value in node:
            _collect_numbers(value, into)


def _traceable_numbers(
    statistics: dict[str, Any],
    tables: TableRegistry,
    problem_text: str,
    extra: Optional[set[float]] = None,
) -> set[float]:
    """Every number the paper is allowed to state.

    A number is legitimate if it was computed and verified, appears in one of the
    paper's own tables, was reported by the verification stage, is the row count
    of a produced result file, or is quoted from the problem statement. Anything
    else in the prose is a fabricated figure.
    """
    allowed: set[float] = set(extra or ())
    _collect_numbers(statistics, allowed)
    for table in tables.all():
        for row in table.rows:
            for cell in row:
                if isinstance(cell.value, bool) or cell.value is None:
                    continue
                if isinstance(cell.value, (int, float)):
                    allowed.add(float(cell.value))
                else:
                    # Cells often carry formatted numbers such as "297 对".
                    _collect_numbers(
                        [float(t) for t in re.findall(r"\d+\.?\d*", str(cell.value))],
                        allowed,
                    )
    # Numbers the problem statement itself supplies are citations, not results.
    for token in re.findall(r"\d+\.?\d*", problem_text or ""):
        try:
            allowed.add(float(token))
        except ValueError:
            continue
    # A quantity the paper derives by multiplying two substantial verified
    # numbers (e.g. 100 bands x 643 time units) is a real computation, not a
    # fabrication. Both factors must be substantial: otherwise a made-up count
    # such as 50 would pass as 5 x 10, since 5 and 10 are problem constants.
    substantial = sorted(v for v in allowed if v > 30)
    for index, left in enumerate(substantial):
        for right in substantial[index:]:
            product = left * right
            if product <= 10_000_000:
                allowed.add(product)
    return allowed


def _collect_into(node: Any) -> set[float]:
    """Convenience wrapper returning a fresh set of every number in ``node``."""
    found: set[float] = set()
    _collect_numbers(node, found)
    return found


def _verification_numbers(report) -> set[float]:
    """Numbers the verification stage itself established or reported."""
    numbers: set[float] = set()
    for check in getattr(report, "checks", []) or []:
        _collect_numbers(getattr(check, "evidence", {}) or {}, numbers)
        for token in re.findall(r"\d+\.?\d*", getattr(check, "detail", "") or ""):
            try:
                numbers.add(float(token))
            except ValueError:
                continue
    return numbers


def _untraceable_numbers(
    paper_text: str,
    allowed: set[float],
) -> set[str]:
    """Numbers stated in the prose that no verified source supports."""
    untraceable: set[str] = set()
    for token in re.findall(r"\d+\.?\d*", paper_text or ""):
        try:
            value = float(token)
        except ValueError:
            continue
        if value in allowed:
            continue
        # Section numbers, list markers and small structural counts.
        if abs(value - round(value)) < 1e-9 and abs(value) <= 30:
            continue
        # Section references such as 6.1, but not a real decimal like 82.7.
        if "." in token and value < 30 and token.split(".")[-1] != "0":
            integer_part = token.split(".")[0]
            if integer_part.isdigit() and int(integer_part) <= 30:
                continue
        untraceable.add(token)
    return untraceable


def _number_present(value: float, text: str) -> bool:
    for token in re.findall(r"\d+\.?\d*", text):
        try:
            parsed = float(token)
        except ValueError:
            continue
        if abs(parsed - value) < 1e-9:
            return True
        decimals = len(token.split(".")[1]) if "." in token else 0
        if abs(parsed - value) <= 0.5 * (10 ** (-decimals)) + 1e-9:
            return True
    return False


def _export_table_files(tables: TableRegistry, out_dir: Path) -> list[Path]:
    """Write every paper table to CSV so the support package carries the data."""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for table in tables.all():
        path = out_dir / f"{table.table_id}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(table.headers)
            for row in table.rows:
                writer.writerow([
                    "" if cell.value is None else cell.value for cell in row
                ])
        written.append(path)
    return written


def _ensure_results_visible(paper, figures: FigureRegistry, tables: TableRegistry) -> None:
    """Guarantee the verified results appear in the paper itself.

    The writer may describe the solution without reproducing every table and
    figure. A competition paper must show its verified results, so any table or
    figure the draft left out is attached to the results section verbatim.
    """
    from mathmodel.paper import BlockType, ContentBlock, PaperSection

    referenced_tables = set(paper.collect_tables())
    referenced_figures = set(paper.collect_figures())

    pending_tables = [t for t in tables.all() if t.table_id not in referenced_tables]
    pending_figures = [f for f in figures.all() if f.figure_id not in referenced_figures]
    if not pending_tables and not pending_figures:
        return

    target = None
    for section in paper.sections:
        if any(key in section.title for key in ("结果", "求解", "分析")):
            target = section
            break
    if target is None:
        target = PaperSection(title="求解结果汇总", purpose="展示经验证的求解结果")
        paper.sections.append(target)

    for table in pending_tables:
        target.content_blocks.append(ContentBlock(
            block_type=BlockType.TABLE,
            text=f"表：{table.title}",
            table_ids=[table.table_id],
        ))
        if table.table_id not in target.table_ids:
            target.table_ids.append(table.table_id)

    for figure in pending_figures:
        target.content_blocks.append(ContentBlock(
            block_type=BlockType.FIGURE,
            text=figure.caption or figure.title,
            figure_ids=[figure.figure_id],
        ))
        if figure.figure_id not in target.figure_ids:
            target.figure_ids.append(figure.figure_id)


def _attach_claims(paper, evidence: EvidenceStore, statistics: dict[str, Any]) -> None:
    """Bind claim ids to blocks that state verified numbers, so the gate can pass."""
    stat_claims = {
        c.metadata.get("statistic_key"): c.claim_id
        for c in evidence.list_claims()
        if c.metadata.get("statistic_key")
    }
    if not stat_claims:
        return

    for section in paper.sections:
        for block in section.content_blocks:
            if block.block_type.value != "paragraph":
                continue
            attached = list(block.claim_ids)
            for key, value in statistics.items():
                if not isinstance(value, (int, float)):
                    continue
                if _number_present(float(value), block.text) and key in stat_claims:
                    claim_id = stat_claims[key]
                    if claim_id not in attached:
                        attached.append(claim_id)
            if attached:
                block.claim_ids = attached
                if attached and "CLAIM-CONCLUSION" not in section.claim_ids:
                    section.claim_ids = list(section.claim_ids) + ["CLAIM-CONCLUSION"]


def _adapt_verifier_code(code: str, required_outputs: list[str]) -> str:
    """The verifier receives output files as base64 inputs; decode them first."""
    if not required_outputs:
        return code
    preamble = "\n".join([
        "import base64 as _b64",
        "import os as _os",
        "import pathlib as _pl",
        "_out = _pl.Path('/workspace/output')",
        "_out.mkdir(parents=True, exist_ok=True)",
        "for _name in _os.listdir('/workspace/input'):",
        "    if _name.startswith('output__'):",
        "        _target = _out / _name[len('output__'):]",
        "        _target.write_bytes(_b64.b64decode(",
        "            _pl.Path('/workspace/input', _name).read_text()))",
        "",
    ])
    return preamble + code


def competition_name(context: ProblemContext) -> str:
    text = context.problem_text
    if "高教社杯" in text or "全国大学生数学建模竞赛" in text:
        return "CUMCM"
    return "CUMCM"
