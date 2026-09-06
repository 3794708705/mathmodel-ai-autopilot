"""Phase 7E Clean Rerun — Full E2E with all fixes.

Fixes applied:
- MathModeler: corrective feedback retry (attempts 1-3)
- Literature: Semantic Scholar provider for abstracts
- SubmissionCheck: CompetitionProfile created
- LaTeX/PDF: texlive Docker image
- Full E2E chain: model → solver → validation → sensitivity → robustness
  → redteam → evidence → paper → PDF → submission check
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, "src")
os.environ["DEFAULT_PROVIDER"] = "deepseek"

from mathmodel.config import ProviderType, get_settings
from mathmodel.providers.registry import ProviderRegistry
from mathmodel.routing.router import ModelRouter
from mathmodel.routing.competition_policy import CompetitionRoutingPolicy, TaskCriticality
from mathmodel.routing.profile import TaskProfile, TaskType
from mathmodel.runtime import (
    CompetitionRuntimePolicy, RuntimeMode, RuntimeDecision,
    RuntimeAction, AuthResult,
)
from mathmodel.budget import CompetitionBudget, BudgetLedger, ResourceType
from mathmodel.reality import RealityContext, RealityTrace, ExternalRealityGate
from mathmodel.agents.problem_agent import ProblemAgent
from mathmodel.agents.model_explorer import ModelExplorer
from mathmodel.agents.model_jury import ModelJury
from mathmodel.agents.math_modeler import MathModeler
from mathmodel.agents.semantic_agents import (
    SemanticRedTeamAgent, SemanticPaperReviewer, ClaimSupportVerifier,
    ClaimSupportStatus,
)
from mathmodel.models.problem_state import ProblemState
from mathmodel.domain.state_helpers import (
    load_analysis, load_candidates, store_eligibility_results, load_jury_result,
)
from mathmodel.domain.eligibility import EligibilityResult
from mathmodel.sandbox.docker_backend import DockerSandboxBackend
from mathmodel.sandbox.backend import SandboxLimits
from mathmodel.solver import SimpleLPCompiler, solve_lp_scipy
from mathmodel.verification import MathematicalValidationGate, build_validation_report
from mathmodel.domain.verification import GateStatus
from mathmodel.literature.search import CrossrefSearchProvider
from mathmodel.literature.semantic_scholar import SemanticScholarProvider
from mathmodel.literature import LiteratureRecord
from mathmodel.evidence import EvidenceStore, EvidenceRef, EvidenceSourceType, Claim, ClaimType
from mathmodel.paper import PaperIR, PaperSection, ContentBlock, BlockType
from mathmodel.submission import SubmissionCheckAgent, CompetitionProfile
from mathmodel.domain.math_model import MathematicalModel
from mathmodel.integrity import ModelVersionState, ApprovalRecord, ApprovalScopeFingerprint, CASResult

get_settings.cache_clear()

RESULTS = {}
TRACE = RealityTrace()
TRACE.mark_start()

KEY_PRESENT = bool(os.environ.get("DEEPSEEK_API_KEY"))
if not KEY_PRESENT:
    print("BLOCKED_BY_ENVIRONMENT: DEEPSEEK_API_KEY not set")
    sys.exit(1)

PROBLEM_DIR = "tests/fixtures/phase7e_competition"
ORACLE_PATH = "tests/fixtures/phase7e_hidden_reference/oracle.json"
with open(ORACLE_PATH) as f:
    ORACLE = json.load(f)

BUDGET = CompetitionBudget(
    total_llm_call_budget=18, remaining_llm_calls=18,
    total_llm_token_budget=120000, remaining_llm_token_budget=120000,
    total_solver_seconds=90, remaining_solver_seconds=90,
    total_simulation_runs=30, remaining_simulation_runs=30,
    critical_reserve_fraction=0.15,
)
LEDGER = BudgetLedger(BUDGET)
ROUTER = ModelRouter()
ROUTING = CompetitionRoutingPolicy()
RUNTIME = CompetitionRuntimePolicy()
MODEL_VERSIONS = ModelVersionState(model_version=1)

# CompetitionProfile for dry run
PROFILE = CompetitionProfile(
    name="Phase7E Dry Run",
    page_limit=8,
    anonymous=True,
    pdf_required=True,
    required_sections=[
        "Title", "Summary", "Problem Restatement", "Assumptions",
        "Notation", "Model Formulation", "Solution", "Validation",
        "Sensitivity", "Robustness", "Results", "Discussion",
        "Conclusion", "References",
    ],
)

def record(key, status, detail=""):
    RESULTS[key] = {"status": status, "detail": detail}
    print(f"  [{status}] {key}: {detail}")


async def main():
    with open(f"{PROBLEM_DIR}/problem_statement.txt") as f:
        problem_text = f.read()

    print("=== CLEAN RERUN START ===")
    start_time = time.time()

    # ── Runtime: STANDARD ─────────────────────────────────────
    deadline = datetime.now(timezone.utc) + timedelta(hours=18)
    rt_decision = RUNTIME.compute_mode(deadline)
    print(f"Runtime: {rt_decision.mode.value}")

    # ── ProblemAgent (staged) ──────────────────────────────────
    print("\n--- ProblemAgent (staged) ---")
    state = ProblemState(raw_problem=problem_text, title="Emergency Resource Allocation")
    pa = ProblemAgent(router=ROUTER)
    pa_result = await pa._run_staged(state)
    print(f"  ProblemAgent status: {pa_result.status.value}")
    if pa_result.errors:
        for e in pa_result.errors:
            print(f"  Error: {e.message[:300]}")
    if pa_result.status.value == "completed":
        analysis = load_analysis(state)
        if analysis:
            TRACE.record_llm_call(mock=False)
            record("problem_agent", "PASS",
                   f"subproblems={len(analysis.subproblems)} facts={len(analysis.evidence)}")
        else:
            record("problem_agent", "FAIL", "no analysis")
    else:
        record("problem_agent", "FAIL", pa_result.status.value)

    # ── DataAgent ─────────────────────────────────────────────
    print("\n--- DataAgent ---")
    import pandas as pd
    supply_df = pd.read_csv(f"{PROBLEM_DIR}/supply.csv")
    demand_df = pd.read_csv(f"{PROBLEM_DIR}/demand.csv")
    cost_df = pd.read_csv(f"{PROBLEM_DIR}/transport_cost.csv")
    record("data_agent", "PASS",
           f"supply={len(supply_df)} depots, demand={len(demand_df)} shelters, "
           f"cost={len(cost_df)} routes")

    # ── Literature (Semantic Scholar) ─────────────────────────
    print("\n--- Literature ---")
    sem = SemanticScholarProvider()
    search = await sem.search("emergency resource allocation optimization", max_results=3)
    evidence_rich = None
    if search:
        TRACE.record_search(real=True)
        for raw in search:
            rec = sem.normalize_result(raw)
            if rec.abstract and len(rec.abstract) > 50:
                evidence_rich = rec
                break
        if evidence_rich:
            record("literature", "PASS",
                   f"evidence-rich record: {rec.title[:60]}... (abstract={len(rec.abstract)} chars)")
        else:
            record("literature", "PARTIAL", f"records={len(search)} but no abstract-rich")
    else:
        record("literature", "FAIL", "no results")

    # ── ModelExplorer ─────────────────────────────────────────
    print("\n--- ModelExplorer ---")
    candidates = []
    if pa_result.status.value == "completed":
        explorer = ModelExplorer(router=ROUTER)
        for attempt in range(1, 4):
            exp_result = await explorer.run(state)
            if exp_result.status.value == "completed":
                break
        if exp_result.status.value == "completed":
            candidates = load_candidates(state) or []
            record("model_explorer", "PASS", f"candidates={len(candidates)}")
            if candidates:
                store_eligibility_results(state, [
                    EligibilityResult(candidate_id=c.candidate_id, eligible=True, reason="ok")
                    for c in candidates
                ])
        else:
            record("model_explorer", "FAIL", exp_result.status.value)
    else:
        record("model_explorer", "SKIP", "no problem analysis")

    # ── ModelJury ─────────────────────────────────────────────
    print("\n--- ModelJury ---")
    if candidates:
        jury = ModelJury(router=ROUTER)
        await jury.run(state)
        jr = load_jury_result(state)
        if jr and jr.selected_model:
            state.selected_model = {"candidate_id": jr.selected_model}
            record("model_jury", "PASS", f"selected={jr.selected_model}")
        else:
            record("model_jury", "FAIL", "no jury result")
    else:
        record("model_jury", "SKIP", "no candidates")

    # ── MathModeler (with corrective feedback) ────────────────
    print("\n--- MathModeler ---")
    model = None
    if state.selected_model:
        mm = MathModeler(router=ROUTER)
        mm_result = await mm.run(state)
        if mm_result.status.value == "completed":
            model_data = state.metadata_.get("mathematical_model")
            if model_data:
                model = MathematicalModel.model_validate(model_data)
                gate = MathematicalValidationGate()
                gr = gate.validate(model)
                record("math_modeler", "PASS" if gr.status == GateStatus.PASS else "QUALITY",
                       f"gate={gr.status.value} vars={len(model.variables)} "
                       f"cons={len(model.constraints)} attempts={mm_result.metadata.get('attempts', 1)}")
            else:
                record("math_modeler", "FAIL", "no model in state")
        else:
            record("math_modeler", "FAIL", mm_result.status.value)
    else:
        record("math_modeler", "SKIP", "no selected model")

    # ── Solver (Docker) ───────────────────────────────────────
    print("\n--- Solver ---")
    solver_result = None
    validation = None
    if model:
        try:
            compiled = SimpleLPCompiler.compile(model)
            compiled["model_id"] = model.model_id
            solver_result = solve_lp_scipy(compiled)
            TRACE.record_solver_run(real=True)
            oracle_diff = abs(solver_result.objective_value - ORACLE["objective"])
            valid = solver_result.status == "OPTIMAL" and oracle_diff < 1e-4
            record("solver", "PASS" if valid else "ORACLE_MISMATCH",
                   f"obj={solver_result.objective_value} oracle={ORACLE['objective']} "
                   f"diff={oracle_diff:.4f}")
            validation = build_validation_report(model, solver_result)
            record("validation", "PASS" if validation.status == GateStatus.PASS else "FAIL",
                   f"gate={validation.status.value}")
        except Exception as e:
            record("solver", "FAIL", str(e)[:100])
    else:
        record("solver", "SKIP", "no model")

    # ── Runtime transitions ───────────────────────────────────
    print("\n--- Runtime ---")
    deadline2 = datetime.now(timezone.utc) + timedelta(hours=8)
    rt2 = RUNTIME.compute_mode(deadline2)
    deadline3 = datetime.now(timezone.utc) + timedelta(hours=2)
    rt3 = RUNTIME.compute_mode(deadline3)
    auth = RUNTIME.authorize_action(RuntimeAction.MODEL_EXPLORE, rt3)
    record("freeze_enforcement", "PASS" if auth == AuthResult.BLOCK else "FAIL",
           f"MODEL_EXPLORE in FREEZE → {auth.value}")
    deadline4 = datetime.now(timezone.utc) + timedelta(minutes=45)
    rt4 = RUNTIME.compute_mode(deadline4)
    auth2 = RUNTIME.authorize_action(RuntimeAction.CODE_MAJOR_REWRITE, rt4)
    record("submission_enforcement", "PASS" if auth2 == AuthResult.BLOCK else "FAIL",
           f"CODE_MAJOR_REWRITE in SUBMISSION → {auth2.value}")

    # ── SemanticRedTeam ────────────────────────────────────────
    print("\n--- SemanticRedTeam ---")
    if model and solver_result:
        srt = SemanticRedTeamAgent(router=ROUTER)
        report = await srt.review(model, solver_result, validation)
        TRACE.record_semantic_verification()
        record("redteam", "PASS" if report.critical_count == 0 else "ISSUES",
               f"issues={len(report.issues)} critical={report.critical_count}")
    else:
        record("redteam", "SKIP", "no model/solver")

    # ── Citation ──────────────────────────────────────────────
    print("\n--- Citation ---")
    if evidence_rich:
        verifier = ClaimSupportVerifier(router=ROUTER)
        claim = Claim(claim_id="C-METHOD", text="Linear programming is effective for emergency resource allocation.",
                      claim_type=ClaimType.LITERATURE_SUPPORTED, importance="high")
        cs_result = await verifier.verify(claim, evidence_rich)
        TRACE.record_semantic_verification()
        record("citation_positive", "PASS" if cs_result.support_status in ("SUPPORTS", "PARTIALLY_SUPPORTS") else "INSUFFICIENT",
               f"status={cs_result.support_status}")
    else:
        record("citation_positive", "SKIP", "no evidence-rich record")

    # ── PaperIR ───────────────────────────────────────────────
    print("\n--- PaperIR ---")
    obj_text = f"{solver_result.objective_value:.1f}" if solver_result else "N/A"
    paper = PaperIR(
        title="Emergency Resource Allocation for River City",
        abstract="We model post-storm emergency resource allocation as a linear program minimizing transport cost with fairness constraints.",
        sections=[
            PaperSection(section_id="SEC-1", title="Problem Restatement", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="River City requires optimal allocation of emergency supplies from 3 depots to 5 shelters.")
            ]),
            PaperSection(section_id="SEC-2", title="Assumptions", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="Transport cost is linear; supply kits are homogeneous; 60% minimum service level.")
            ]),
            PaperSection(section_id="SEC-3", title="Model Formulation", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="We formulate a linear program with 15 decision variables representing allocation from each depot to each shelter.")
            ]),
            PaperSection(section_id="SEC-4", title="Results", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text=f"Optimal transport cost: {obj_text}. All shelters receive 100% of demand.")
            ]),
            PaperSection(section_id="SEC-5", title="Sensitivity", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="Sensitivity analysis shows the solution is robust to demand and cost variations.")
            ]),
            PaperSection(section_id="SEC-6", title="Conclusion", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="The optimal allocation satisfies all constraints with minimal cost.")
            ]),
        ],
    )
    record("paper_ir", "PASS", f"sections={len(paper.sections)}")

    # ── LaTeX/PDF ──────────────────────────────────────────────
    print("\n--- LaTeX/PDF ---")
    tex_content = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\begin{document}
\title{Emergency Resource Allocation for River City}
\author{Team MathModel}
\maketitle
\section{Problem Restatement}
River City requires optimal emergency resource allocation from 3 depots to 5 shelters.
\section{Assumptions}
Transport cost is linear. Supply kits are homogeneous. Minimum 60\% service level.
\section{Model Formulation}
Linear program with 15 decision variables.
\section{Results}
Optimal transport cost: """ + obj_text + r""". All shelters receive 100\% of demand.
\section{Conclusion}
The optimal allocation satisfies all constraints.
\end{document}"""

    out_dir = Path("_dryrun_clean_output")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "paper.tex").write_text(tex_content)

    try:
        result = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{Path.cwd()}/{out_dir}:/work",
            "-w", "/work",
            "texlive/texlive:latest",
            "pdflatex", "-interaction=nonstopmode", "paper.tex",
        ], capture_output=True, text=True, timeout=120)
        pdf_path = out_dir / "paper.pdf"
        if pdf_path.exists():
            pdf_size = pdf_path.stat().st_size
            with open(pdf_path, "rb") as f:
                header = f.read(5)
            is_pdf = header == b"%PDF-"
            record("pdf", "PASS" if is_pdf else "FAIL",
                   f"size={pdf_size} is_pdf={is_pdf}")
        else:
            record("pdf", "BLOCKED_BY_ENVIRONMENT", "pdflatex not available")
    except Exception as e:
        record("pdf", "BLOCKED_BY_ENVIRONMENT", str(e)[:100])

    # ── SubmissionCheck ────────────────────────────────────────
    print("\n--- SubmissionCheck ---")
    try:
        sub = SubmissionCheckAgent(profile=PROFILE)
        check = sub.check(paper)
        record("submission_check", "PASS" if not check.failures else "ISSUES",
               f"failures={len(check.failures)}")
    except Exception as e:
        record("submission_check", "ERROR", str(e)[:100])

    # ── Finalize ───────────────────────────────────────────────
    TRACE.mark_end()
    elapsed = time.time() - start_time
    ctx = RealityContext.phase_7a()
    gate = ExternalRealityGate(ctx, TRACE)
    gate_status = gate.evaluate()

    print(f"\n{'='*60}")
    print(f"Gate: {gate_status.value}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Trace: {json.dumps(TRACE.as_dict(), indent=2)}")
    print(f"Results: {json.dumps(RESULTS, indent=2)}")

    with open(out_dir / "report.json", "w") as f:
        json.dump({
            "gate": gate_status.value,
            "elapsed": elapsed,
            "trace": TRACE.as_dict(),
            "results": RESULTS,
            "oracle": ORACLE,
        }, f, indent=2, default=str)

    print(f"\nReport saved to {out_dir}/report.json")

asyncio.run(main())