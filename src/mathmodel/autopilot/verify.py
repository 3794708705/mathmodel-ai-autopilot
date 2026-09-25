"""MathModel AI — CUMCM Autopilot: verification gate.

Nothing reaches the paper until it passes verification here.

Two independent layers:
  1. Deterministic host checks — real files, real parsing, real recomputation.
  2. An independently generated verification program, executed in the same
     sandbox, that must recompute the answer without seeing the solver code.

A "the program ran" result is never treated as "the model is correct".
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from mathmodel.autopilot.codegen import (
    CODE_MAX_TOKENS,
    GeneratedProgram,
    SandboxProgramRunner,
    _extract_code,
    _extract_header_field,
    _parse_summary,
    generate_program_source,
)
from mathmodel.domain.math_model import MathematicalModel
from mathmodel.routing.profile import TaskProfile, TaskType
from mathmodel.routing.router import ModelRouter

IDENTIFIER_RE = re.compile(r"^[A-Za-z]{1,4}\d{2,6}$")

# A key describing what is LEFT OVER after solving, as opposed to what was
# found: "residual_conflicts" is a failure, "total_conflict_pairs" is a count.
_RESIDUAL_MARKERS = (
    "residual", "remaining", "unresolved", "leftover", "left_over", "final", "after",
)
_VIOLATION_MARKERS = ("conflict", "violation", "collision", "overlap", "infeasib")


def _self_reported_leftovers(statistics: dict[str, Any]) -> Optional[dict[str, float]]:
    """Residual-violation counters the solver itself reported.

    Returns ``None`` when the program reported no such counter at all, an empty
    dict when every counter it did report is zero, and a mapping of
    ``dotted.path -> value`` for the non-zero ones.
    """
    found: dict[str, float] = {}
    saw_any = False

    def walk(node: Any, path: str) -> None:
        nonlocal saw_any
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
            return
        if isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
            return

        leaf = path.rsplit(".", 1)[-1].lower()
        if not any(marker in leaf for marker in _RESIDUAL_MARKERS):
            return
        if not any(marker in leaf for marker in _VIOLATION_MARKERS):
            return
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            return
        saw_any = True
        if abs(float(node)) > 1e-9:
            found[path] = float(node)

    walk(statistics, "")
    return found if saw_any else None
COUNT_KEY_RE = re.compile(r"(count|num|total|数量|总数|个数)", re.IGNORECASE)


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    BLOCKED = "BLOCKED"


class VerificationCheck(BaseModel):
    check_id: str = Field(default_factory=lambda: f"VC-{uuid4().hex[:8]}")
    name: str
    category: str = "general"
    status: CheckStatus = CheckStatus.NOT_RUN
    detail: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class VerificationReport(BaseModel):
    report_id: str = Field(default_factory=lambda: f"VER-{uuid4().hex[:8]}")
    model_id: str = ""
    model_version: Optional[int] = None
    solver_run_id: str = ""
    execution_run_id: str = ""
    checks: list[VerificationCheck] = Field(default_factory=list)
    blocking_failures: list[str] = Field(default_factory=list)
    overall: CheckStatus = CheckStatus.NOT_RUN
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passed(self) -> bool:
        return self.overall == CheckStatus.PASS

    def failed_checks(self) -> list[VerificationCheck]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    def to_prompt_feedback(self) -> list[str]:
        """Failed checks as repair instructions, counterexamples included.

        A bare "39 conflicts remain" tells the solver nothing it can act on.
        The concrete offending items do, so any counterexamples the verifier
        reported are carried into the feedback.
        """
        lines: list[str] = []
        for check in self.failed_checks():
            lines.append(f"[{check.category}] {check.name}: {check.detail}")
            evidence = check.evidence or {}
            counterexamples = evidence.get("counterexamples")
            if counterexamples:
                rendered = json.dumps(counterexamples, ensure_ascii=False)
                lines.append(
                    f"    concrete counterexamples from the check: {rendered[:1500]}"
                )
            elif evidence:
                rendered = json.dumps(evidence, ensure_ascii=False)
                lines.append(f"    evidence: {rendered[:800]}")
        return lines


# ═══════════════════════════════════════════════════════════════
# Deterministic host-side checks
# ═══════════════════════════════════════════════════════════════

def read_output_table(path: Path) -> tuple[list[str], list[list[Any]]]:
    """Read an output artifact as (headers, rows). Real parse, no guessing."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb[wb.sheetnames[0]]
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
        finally:
            wb.close()
    elif path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            rows = [list(r) for r in csv.reader(handle)]
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")

    rows = [r for r in rows if any(v is not None and str(v).strip() != "" for v in r)]
    if not rows:
        return [], []
    headers = [("" if h is None else str(h).strip()) for h in rows[0]]
    return headers, rows[1:]


def collect_identifier_pool(data_dir: Path) -> set[str]:
    """Every identifier-like token that exists in the real input data."""
    pool: set[str] = set()
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() not in (".csv", ".xlsx"):
            continue
        try:
            _, rows = read_output_table(path)
        except Exception:
            continue
        for row in rows:
            for cell in row:
                if cell is None:
                    continue
                token = str(cell).strip()
                if IDENTIFIER_RE.match(token):
                    pool.add(token)
    return pool


class DeterministicVerifier:
    """Host-side checks that never depend on the solver's own claims."""

    def __init__(self, data_dir: str | Path):
        self._data_dir = Path(data_dir)

    def run(
        self,
        outcome,
        output_dir: str | Path,
        required_outputs: list[str],
        template_headers: dict[str, list[str]],
        expected_row_counts: Optional[dict[str, int]] = None,
    ) -> list[VerificationCheck]:
        output_dir = Path(output_dir)
        checks: list[VerificationCheck] = []

        # 1. Execution really happened
        if not outcome.execution_real:
            checks.append(VerificationCheck(
                name="execution_was_real",
                category="execution",
                status=CheckStatus.FAIL,
                detail="Solver result did not come from real sandbox execution",
                evidence={"status": outcome.status, "sandbox": outcome.sandbox},
            ))
        elif not outcome.succeeded:
            checks.append(VerificationCheck(
                name="execution_was_real",
                category="execution",
                status=CheckStatus.FAIL,
                detail=f"Sandbox execution status={outcome.status} exit={outcome.exit_code}",
                evidence={"stderr": outcome.stderr[-2000:]},
            ))
        else:
            checks.append(VerificationCheck(
                name="execution_was_real",
                category="execution",
                status=CheckStatus.PASS,
                detail=f"exit=0 runtime={outcome.runtime_seconds}s image={outcome.sandbox.get('image')}",
                evidence={"code_hash": outcome.sandbox.get("code_hash")},
            ))

        # 2. Required outputs exist, are non-empty, and parse
        parsed: dict[str, tuple[list[str], list[list[Any]]]] = {}
        missing: list[str] = []
        unreadable: list[str] = []
        empty: list[str] = []
        for name in required_outputs:
            path = output_dir / name
            if not path.exists():
                missing.append(name)
                continue
            if path.stat().st_size == 0:
                empty.append(name)
                continue
            try:
                parsed[name] = read_output_table(path)
            except Exception as exc:
                unreadable.append(f"{name}: {exc}")

        if missing or empty or unreadable:
            checks.append(VerificationCheck(
                name="required_outputs_present",
                category="outputs",
                status=CheckStatus.FAIL,
                detail=(
                    f"missing={missing} empty={empty} unreadable={unreadable}"
                ),
                evidence={"required": required_outputs},
            ))
        else:
            checks.append(VerificationCheck(
                name="required_outputs_present",
                category="outputs",
                status=CheckStatus.PASS,
                detail="; ".join(
                    f"{name}: {len(rows)} rows" for name, (_, rows) in parsed.items()
                ),
                evidence={
                    "row_counts": {n: len(r) for n, (_, r) in parsed.items()},
                },
            ))

        # 3. Output structure matches the provided template
        structure_issues: list[str] = []
        compared = 0
        for name, (headers, _) in parsed.items():
            expected = template_headers.get(name)
            if not expected:
                continue
            compared += 1
            expected_norm = [h.strip().lower() for h in expected if h.strip()]
            actual_norm = [h.strip().lower() for h in headers]
            if len(actual_norm) < len(expected_norm):
                structure_issues.append(
                    f"{name}: expected >= {len(expected_norm)} columns, got {len(actual_norm)}"
                )
                continue
            # The supplied template's header row is part of the answer, so the
            # wording has to match. Comparing only column counts let a result
            # file pass while every heading differed from the template.
            mismatched = [
                f"column {index + 1}: expected {expected_norm[index]!r}, "
                f"got {actual_norm[index]!r}"
                for index in range(len(expected_norm))
                if actual_norm[index] != expected_norm[index]
            ]
            if mismatched:
                structure_issues.append(
                    f"{name}: header does not match the template — "
                    + "; ".join(mismatched[:4])
                )
        if compared == 0:
            checks.append(VerificationCheck(
                name="output_structure_matches_template",
                category="outputs",
                status=CheckStatus.NOT_APPLICABLE,
                detail="No template files were supplied",
            ))
        elif structure_issues:
            checks.append(VerificationCheck(
                name="output_structure_matches_template",
                category="outputs",
                status=CheckStatus.FAIL,
                detail="; ".join(structure_issues),
            ))
        else:
            checks.append(VerificationCheck(
                name="output_structure_matches_template",
                category="outputs",
                status=CheckStatus.PASS,
                detail=f"{compared} output file(s) match the supplied template shape",
            ))

        # 3b. A template keyed by plan ID must be answered for every plan.
        # Listing only the plans that were changed leaves the reader unable to
        # see the disposition of the rest, so the row count has to cover the data.
        if not expected_row_counts:
            checks.append(VerificationCheck(
                name="per_plan_table_covers_every_plan",
                category="outputs",
                status=CheckStatus.NOT_APPLICABLE,
                detail="No per-plan template was supplied",
            ))
        else:
            coverage_issues = []
            for name, required_rows in expected_row_counts.items():
                entry = parsed.get(name)
                if entry is None:
                    continue
                written = len(entry[1])
                if written < required_rows:
                    coverage_issues.append(
                        f"{name}: {written} row(s) for {required_rows} plan(s)"
                    )
            checks.append(VerificationCheck(
                name="per_plan_table_covers_every_plan",
                category="outputs",
                status=CheckStatus.FAIL if coverage_issues else CheckStatus.PASS,
                detail=(
                    "; ".join(coverage_issues)
                    + " — a table keyed by 用频装备编号 must give one row per plan, "
                    "marking unchanged plans as not cancelled"
                    if coverage_issues
                    else f"{len(expected_row_counts)} per-plan table(s) cover every plan"
                ),
                evidence={"expected_rows": dict(expected_row_counts)},
            ))

        # 4. Referential integrity against the real input data.
        # A 序号 column numbers newly created items, so those tokens cannot
        # appear in the input and flagging them would reject a correct answer.
        # Every other column is treated as a reference to an existing entity.
        pool = collect_identifier_pool(self._data_dir)
        unknown: dict[str, list[str]] = {}
        if pool:
            for name, (headers, rows) in parsed.items():
                referencing = [
                    index for index, header in enumerate(headers)
                    if "序号" not in header
                ]
                if not referencing:
                    continue
                bad: list[str] = []
                for row in rows:
                    for index in referencing:
                        if index >= len(row) or row[index] is None:
                            continue
                        token = str(row[index]).strip()
                        if IDENTIFIER_RE.match(token) and token not in pool:
                            bad.append(token)
                if bad:
                    unknown[name] = sorted(set(bad))[:20]
        if not pool:
            checks.append(VerificationCheck(
                name="referential_integrity",
                category="data",
                status=CheckStatus.NOT_APPLICABLE,
                detail="No identifier-like tokens found in the input data",
            ))
        elif unknown:
            checks.append(VerificationCheck(
                name="referential_integrity",
                category="data",
                status=CheckStatus.FAIL,
                detail=f"Outputs reference identifiers absent from input data: {unknown}",
                evidence={"identifier_pool_size": len(pool)},
            ))
        else:
            checks.append(VerificationCheck(
                name="referential_integrity",
                category="data",
                status=CheckStatus.PASS,
                detail=f"All output identifiers exist in the input data (pool={len(pool)})",
            ))

        # 5. The program reported machine-readable statistics
        stats = (outcome.summary or {}).get("statistics")
        if not isinstance(stats, dict) or not stats:
            checks.append(VerificationCheck(
                name="statistics_reported",
                category="outputs",
                status=CheckStatus.FAIL,
                detail="Program did not print a non-empty 'statistics' JSON object",
                evidence={"summary_keys": sorted((outcome.summary or {}).keys())},
            ))
        else:
            checks.append(VerificationCheck(
                name="statistics_reported",
                category="outputs",
                status=CheckStatus.PASS,
                detail=f"{len(stats)} statistic(s) reported",
                evidence={"statistics": stats},
            ))

        # 6. The program's own report must not admit that the answer is still
        #    infeasible. A solver that says "291 conflicts remain" has not
        #    solved the problem, and no downstream check would otherwise notice.
        leftovers = _self_reported_leftovers(stats if isinstance(stats, dict) else {})
        if leftovers is None:
            checks.append(VerificationCheck(
                name="no_self_reported_residual_violations",
                category="correctness",
                status=CheckStatus.NOT_APPLICABLE,
                detail="Program reported no residual/remaining violation counters",
            ))
        elif leftovers:
            checks.append(VerificationCheck(
                name="no_self_reported_residual_violations",
                category="correctness",
                status=CheckStatus.FAIL,
                detail=(
                    "The solver's own statistics report unresolved violations, so the "
                    "produced solution is not valid: "
                    + "; ".join(f"{path}={value}" for path, value in leftovers.items())
                ),
                evidence={"residual": leftovers},
            ))
        else:
            checks.append(VerificationCheck(
                name="no_self_reported_residual_violations",
                category="correctness",
                status=CheckStatus.PASS,
                detail="All self-reported residual violation counters are zero",
            ))

        return checks


# ═══════════════════════════════════════════════════════════════
# Independent verification program
# ═══════════════════════════════════════════════════════════════

class IndependentCheck(BaseModel):
    name: str
    status: str = "FAIL"
    detail: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class IndependentVerificationProgram(BaseModel):
    approach: str = ""
    code: str = Field(..., min_length=1)


class IndependentVerifier:
    """Generates and runs a verifier that never sees the solver's code."""

    def __init__(self, router: ModelRouter, runner: SandboxProgramRunner):
        self._router = router
        self._runner = runner

    async def generate(
        self,
        model: MathematicalModel,
        problem_text: str,
        required_outputs: list[str],
        solver_statistics: dict[str, Any],
        available_inputs: list[str],
        failure_feedback: Optional[list[str]] = None,
        data_schema: str = "",
    ) -> IndependentVerificationProgram:
        prompt = "\n".join([
            "Write ONE self-contained Python program that INDEPENDENTLY VERIFIES a "
            "solution to a mathematical modeling competition problem.",
            "",
            "You are an adversarial verifier. Your job is to try to falsify the "
            "claimed results by recomputing them from the raw data yourself.",
            "",
            "## PROBLEM STATEMENT (excerpt)",
            problem_text[:10000],
            "",
            *(
                [
                    "## INPUT DATA SCHEMA (the columns that really exist)",
                    "This is the exact shape of the input data, parsed from the real "
                    "attachment. Its columns are authoritative: do not assume any "
                    "extra column exists, and do not require more non-empty values "
                    "per row than there are columns. A record that fails your parse "
                    "is a bug in your parser, not a property of the data.",
                    data_schema[:8000],
                    "",
                ]
                if data_schema
                else []
            ),
            "## MODEL THAT WAS USED (for understanding the claims only)",
            json.dumps(
                {
                    "name": model.name,
                    "description": model.description,
                    "objectives": [o.expression for o in model.objectives],
                    "constraints": [c.expression for c in model.constraints],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "## CLAIMED STATISTICS (from the solver program — verify, do not trust)",
            json.dumps(solver_statistics, ensure_ascii=False, indent=2),
            "",
            "## FILES AVAILABLE",
            f"- /workspace/input/ contains: {available_inputs}",
            f"- /workspace/output/ contains the solution files: {required_outputs}",
            "",
            "## FILE HANDLING (mandatory)",
            "- Never hardcode a file name you were not given. Discover the actual "
            "files at runtime, e.g. `sorted(os.listdir('/workspace/input'))`, and "
            "pick the one you need from that listing.",
            "- If a file you expect is missing, report the check as NOT_APPLICABLE "
            "with the listing you observed in `detail`. Do not crash, and do not "
            "invent a path.",
            "",
            "## WHAT TO CHECK (implement each one)",
            "1. `independent_recomputation` — recompute the claimed statistics from "
            "the raw input data with your own code, and compare against the claimed "
            "values. Report the recomputed values in `evidence`.",
            "2. `constraint_satisfaction` — check every hard constraint of the problem "
            "against the solution files. Report the worst violation found. Put the "
            "concrete offending items in `evidence.counterexamples` as a list of short "
            "strings (for example the specific conflicting pairs or out-of-range "
            "entries, up to 20 of them). Naming the exact items is what lets the "
            "solver repair its program.",
            "2b. `constraint_<n>` — emit ONE ADDITIONAL CHECK PER CONSTRAINT listed "
            "in the model above, named `constraint_1`, `constraint_2`, ... in that "
            "order, and check that constraint against the solution files directly. "
            "Constraints are usually inequalities on the adjusted values; compare "
            "each adjusted value against the ORIGINAL value from the input data and "
            "flag every entry that breaks the bound. Report the offenders in "
            "`evidence.counterexamples`. A constraint you did not check is a "
            "constraint the solver is free to break, so do not skip any. Every "
            "constraint check MUST report `evidence` describing what it examined "
            "(for example the number of entries checked and the worst offender); "
            "a PASS with empty `evidence` is treated as NOT RUN.",
            "3. `feasibility` — check the solution is physically/logically valid "
            "(no impossible values, no duplicate assignments, ranges respected).",
            "4. `small_case_check` — construct a small instance where the correct "
            "answer is known by construction, run the same reasoning on it, and "
            "confirm it gives the known answer.",
            "5. `baseline_comparison` — compute a simple baseline (e.g. no "
            "optimization / trivial assignment) and report whether the claimed "
            "solution is at least as good. Use NOT_APPLICABLE if no baseline exists.",
            "",
            "6. `objectives` — for EVERY required output file, state whether the "
            "solution actually achieves that file's purpose, judged by reading the "
            "file and recomputing the quantity it is supposed to optimise or "
            "eliminate. Do not judge an output by whether it is well-formed, and do "
            "not substitute properties of the INPUT data (ranges, types, counts) for "
            "a verdict on the answer. A file that is structurally valid but leaves "
            "the problem unsolved has `achieved: false`.",
            "",
            "## EXECUTION CONTRACT",
            "- Runs as `python /workspace/code/code.py`, cwd=/workspace.",
            "- numpy, scipy, pandas, openpyxl, matplotlib available. No network.",
            "- Read only from /workspace/input and /workspace/output.",
            "- The LAST line of stdout must be a single JSON object:",
            '  {"checks": [{"name": "...", "status": "PASS|FAIL|NOT_APPLICABLE", '
            '"detail": "...", "evidence": {...}}], '
            '"input_records": <int>, '
            '"objectives": [{"output": "result1.xlsx", "achieved": true, '
            '"evidence": "..."}], '
            '"statistics": {...}, "overall": "PASS|FAIL"}',
            "- `objectives` is REQUIRED and must contain exactly one entry per "
            "required output file listed above. `achieved` must be your own verdict "
            "after reading that file, with the recomputed quantity in `evidence`.",
            "- `input_records` is REQUIRED: the number of data records you actually "
            "parsed from the input attachment. If you parsed none, say 0 — a verdict "
            "produced without reading the data is worthless and will be rejected.",
            "- `statistics` is REQUIRED and must contain the values you recomputed "
            "yourself, keyed by the SAME names used in CLAIMED STATISTICS above, so "
            "they can be compared automatically. Recompute at least the top-level "
            "numeric claims (for example a total count). If you truly cannot "
            "recompute a value, omit that key rather than copying the claimed one.",
            "",
            "## HARD RULES",
            "- Recompute independently. Do NOT simply re-read the claimed numbers and "
            "echo them back as PASS.",
            "- If a check cannot be performed, use status NOT_APPLICABLE with the "
            "reason in `detail` — never report PASS for something you did not check.",
            "- Be adversarial: a wrong solution should FAIL.",
            "",
            "## OUTPUT FORMAT (follow exactly)",
            "Reply with raw Python source only. No markdown fences, no commentary "
            "before or after the code. The first line must be a comment:",
            "# APPROACH: <one line describing how you verified the solution>",
            "",
            "Keep the program under about 250 lines. A complete, compact program "
            "that reaches the final JSON line is worth far more than an exhaustive "
            "one that is cut off mid-file.",
        ])

        if failure_feedback:
            prompt += "\n".join([
                "",
                "## YOUR PREVIOUS VERIFIER PROGRAM CRASHED — FIX IT",
                *[f"  - {item}" for item in failure_feedback],
                "",
                "Write the corrected complete program. Make it defensive: guard "
                "every indexing, slicing and attribute access against empty or "
                "missing data, and never assume a value is present. The program "
                "must always reach the final JSON line even if a check cannot be "
                "completed — report NOT_APPLICABLE instead of crashing.",
            ])

        profile = TaskProfile.for_task_type(TaskType.VALIDATION)
        # Requested as raw source, like the solver: a whole program cannot be
        # reliably returned inside a JSON string field.
        code = await generate_program_source(
            self._router,
            profile=profile,
            prompt=prompt,
            system_prompt=(
                "You are a rigorous, adversarial verifier for mathematical modeling "
                "results. You independently recompute claims and you report FAIL when "
                "evidence does not support a claim. You never rubber-stamp. "
                "You reply with raw Python source only."
            ),
        )
        if not code.strip():
            raise ValueError(
                "Independent verifier generation returned no program"
            )
        return IndependentVerificationProgram(
            approach=_extract_header_field(code, "APPROACH")
            or "adversarial independent re-computation",
            code=code,
        )

    async def run(
        self,
        program: IndependentVerificationProgram,
        input_files: dict[str, str],
        output_dir: str | Path,
    ):
        runnable = GeneratedProgram(
            approach=program.approach,
            code=program.code,
            dependencies=[],
            expected_outputs=[],
        )
        return await self._runner.run(runnable, input_files, output_dir)


def _objective_check(
    summary: dict[str, Any], required_outputs: list[str]
) -> VerificationCheck:
    """Require an explicit per-output verdict that the file answers its question.

    The verifier must report `objectives`, one entry per required output:
    `{"output": "<filename>", "achieved": true|false, "evidence": "..."}`.
    """
    raw = summary.get("objectives")
    if not isinstance(raw, list) or not raw:
        return VerificationCheck(
            name="independent_objectives_achieved",
            category="independent",
            status=CheckStatus.FAIL,
            detail=(
                "Independent verifier did not report an `objectives` list, so it "
                "never stated whether the solution actually achieves each output's "
                "purpose. Checking only properties of the input data does not "
                "establish that the answer is correct."
            ),
            evidence={"required_outputs": required_outputs},
        )

    seen: dict[str, dict[str, Any]] = {}
    for item in raw:
        if isinstance(item, dict) and item.get("output"):
            seen[str(item["output"]).strip()] = item

    missing = [name for name in required_outputs if name not in seen]
    failed = [
        name for name in required_outputs
        if name in seen and seen[name].get("achieved") is not True
    ]
    if missing or failed:
        return VerificationCheck(
            name="independent_objectives_achieved",
            category="independent",
            status=CheckStatus.FAIL,
            detail=(
                f"outputs with no verdict: {missing}; "
                f"outputs whose purpose was not achieved: "
                f"{ {n: seen[n].get('evidence', '') for n in failed} }"
            ),
            evidence={"objectives": raw},
        )
    return VerificationCheck(
        name="independent_objectives_achieved",
        category="independent",
        status=CheckStatus.PASS,
        detail=f"Every required output ({len(required_outputs)}) was confirmed to achieve its purpose",
        evidence={"objectives": raw},
    )


def checks_from_independent_summary(
    summary: dict[str, Any],
    required_outputs: Optional[list[str]] = None,
) -> tuple[list[VerificationCheck], dict[str, Any]]:
    """Turn the verifier program's JSON summary into VerificationChecks."""
    checks: list[VerificationCheck] = []
    raw_checks = summary.get("checks")
    if not isinstance(raw_checks, list) or not raw_checks:
        return (
            [VerificationCheck(
                name="independent_verifier_reported",
                category="independent",
                status=CheckStatus.FAIL,
                detail="Independent verifier did not report any checks",
                evidence={"summary": summary},
            )],
            {},
        )

    # A verifier that could not read its inputs still tends to report every
    # check as PASS, because each check is a no-op over an empty collection.
    # That is a vacuous verdict, not evidence, so require the verifier to state
    # how many input records it actually parsed.
    records = summary.get("input_records")
    if not isinstance(records, int) or records <= 0:
        checks.append(VerificationCheck(
            name="independent_read_the_input",
            category="independent",
            status=CheckStatus.FAIL,
            detail=(
                "Independent verifier reported "
                + ("no `input_records` field" if records is None
                   else f"input_records={records}")
                + ", so it parsed none of the input data. Its PASS results cover "
                "nothing and cannot be trusted."
            ),
            evidence={"input_records": records},
        ))
    else:
        checks.append(VerificationCheck(
            name="independent_read_the_input",
            category="independent",
            status=CheckStatus.PASS,
            detail=f"Independent verifier parsed {records} input record(s)",
            evidence={"input_records": records},
        ))

    # Every required output file must be confronted by name. A verifier is free
    # to spend all its checks on properties of the INPUT data and never ask
    # whether the solution answers the question — observed in practice as a
    # report whose checks were all input ranges while every conflict in the
    # answer was still unresolved. Requiring one explicit verdict per output
    # makes that failure mode impossible to pass.
    if required_outputs:
        checks.append(_objective_check(summary, required_outputs))

    for item in raw_checks:
        if not isinstance(item, dict):
            continue
        status_text = str(item.get("status", "FAIL")).strip().upper()
        try:
            status = CheckStatus(status_text)
        except ValueError:
            status = CheckStatus.FAIL
        name = str(item.get("name", "check"))
        detail = str(item.get("detail", ""))[:2000]
        evidence = (
            item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
        )
        # A per-constraint check that reports PASS with nothing to show for it
        # has almost certainly not run: it is a loop over an empty or unfiltered
        # collection. Such a verdict must not count as coverage, so it is
        # downgraded to NOT_RUN, which keeps the report from passing.
        if (
            status == CheckStatus.PASS
            and name.startswith("constraint")
            and not evidence
        ):
            status = CheckStatus.NOT_RUN
            detail = (
                detail
                + " [no evidence reported, so this constraint was not actually "
                "checked]"
            )
        checks.append(VerificationCheck(
            name=f"independent_{name}",
            category="independent",
            status=status,
            detail=detail,
            evidence=evidence,
        ))
    return checks, summary


def _compare_statistic_values(
    claimed: Any,
    recomputed: Any,
    path: str,
    tolerance: float,
    mismatches: list[str],
    counter: list[int],
) -> None:
    """Compare two statistic values, recursing into nested mappings.

    Only values present in BOTH reports are compared. Two programs naturally
    report different extra keys, and whole-dict equality would flag those as
    disagreement even when every shared number matches exactly.
    """
    if isinstance(claimed, dict) and isinstance(recomputed, dict):
        for key, value in claimed.items():
            if key in recomputed:
                _compare_statistic_values(
                    value, recomputed[key], f"{path}.{key}" if path else str(key),
                    tolerance, mismatches, counter,
                )
        return

    # bool is a subclass of int; comparing True with 1 would be wrong here.
    if isinstance(claimed, bool) or isinstance(recomputed, bool):
        counter[0] += 1
        if claimed != recomputed:
            mismatches.append(f"{path}: solver={claimed} independent={recomputed}")
        return

    if isinstance(claimed, (int, float)) and isinstance(recomputed, (int, float)):
        counter[0] += 1
        if abs(float(claimed) - float(recomputed)) > tolerance:
            mismatches.append(f"{path}: solver={claimed} independent={recomputed}")
        return

    counter[0] += 1
    if claimed != recomputed:
        mismatches.append(f"{path}: solver={claimed!r} independent={recomputed!r}")


def compare_statistics(
    solver_stats: dict[str, Any],
    independent_stats: dict[str, Any],
    tolerance: float = 1e-6,
) -> VerificationCheck:
    """Compare numbers reported by two independent programs."""
    if not independent_stats:
        return VerificationCheck(
            name="independent_statistics_agree",
            category="independent",
            status=CheckStatus.NOT_RUN,
            detail="Independent verifier did not report comparable statistics",
        )

    mismatches: list[str] = []
    counter = [0]
    for key, claimed in solver_stats.items():
        if key not in independent_stats:
            continue
        _compare_statistic_values(
            claimed, independent_stats[key], str(key), tolerance, mismatches, counter,
        )
    compared = counter[0]

    if compared == 0:
        return VerificationCheck(
            name="independent_statistics_agree",
            category="independent",
            status=CheckStatus.NOT_RUN,
            detail="No overlapping statistic keys between solver and verifier",
            evidence={"solver_keys": sorted(solver_stats), "independent_keys": sorted(independent_stats)},
        )
    if mismatches:
        return VerificationCheck(
            name="independent_statistics_agree",
            category="independent",
            status=CheckStatus.FAIL,
            detail="; ".join(mismatches[:10]),
            evidence={"compared": compared},
        )
    return VerificationCheck(
        name="independent_statistics_agree",
        category="independent",
        status=CheckStatus.PASS,
        detail=f"{compared} statistic(s) independently reproduced",
        evidence={"compared": compared},
    )


def build_report(
    model: MathematicalModel,
    execution_run_id: str,
    checks: list[VerificationCheck],
) -> VerificationReport:
    report = VerificationReport(
        model_id=model.model_id,
        model_version=model.version,
        solver_run_id=execution_run_id,
        execution_run_id=execution_run_id,
        checks=checks,
    )

    blocking = [
        f"[{c.category}] {c.name}: {c.detail}"
        for c in checks
        if c.status in (CheckStatus.FAIL, CheckStatus.BLOCKED)
    ]
    report.blocking_failures = blocking

    if blocking:
        report.overall = CheckStatus.FAIL
    elif any(c.status == CheckStatus.NOT_RUN for c in checks):
        report.overall = CheckStatus.FAIL
    else:
        report.overall = CheckStatus.PASS
    return report
