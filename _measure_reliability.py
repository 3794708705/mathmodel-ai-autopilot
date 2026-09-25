"""Measure end-to-end convergence reliability of the CUMCM Autopilot.

Fixed model, fixed problem, fixed configuration; only the run seed varies.
Each run is a complete pipeline (intake -> final_check) in its own run
directory. Outcomes are appended to a JSONL file as they settle, so a partial
measurement survives an interruption.

Usage:
    python _measure_reliability.py --runs 6 --parallel 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _user_env(name: str) -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"[System.Environment]::GetEnvironmentVariable('{name}','User')"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return (out.stdout or "").strip()
    except Exception:
        return ""


# ── Fixed configuration (identical for every run) ────────────────────────
MODEL = os.environ.get("MATHMODEL_LOCAL_MODEL", "global:deepseek-v4.1-flash")
os.environ["OPENAI_API_KEY"] = (
    os.environ.get("MATHMODEL_LOCAL_7863_API_KEY") or _user_env("MATHMODEL_LOCAL_7863_API_KEY")
)
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:7863/v1"
os.environ["OPENAI_DEFAULT_MODEL"] = MODEL
os.environ["DEFAULT_PROVIDER"] = "openai"

sys.path.insert(0, str(ROOT / "src"))

from mathmodel.config import get_settings  # noqa: E402

get_settings.cache_clear()

from mathmodel.autopilot import CUMCMAutopilot  # noqa: E402

CASE = ROOT / ".tmp" / "case_dti"
FILES = [
    CASE / "D题.pdf",
    CASE / "附件1.xlsx",
    CASE / "result1.xlsx",
    CASE / "result2.xlsx",
    CASE / "result3.xlsx",
    CASE / "result4.xlsx",
]

WORKSPACE = ROOT / "runs" / "reliability"
TAG = os.environ.get("MATHMODEL_MEASURE_TAG", "")
RECORDS = ROOT / ".tmp" / f"reliability_records{TAG}.jsonl"
LOGS = ROOT / ".tmp" / f"reliability_logs{TAG}"


def _categorise(blockers: list[str]) -> list[str]:
    """Group blocker text into stable failure-mode labels."""
    labels = []
    for b in blockers:
        low = b.lower()
        if "read_the_input" in low or "input_records=0" in low:
            labels.append("verifier_read_no_input")
        elif "independent_verifier" in low and ("crash" in low or "failed to run" in low):
            labels.append("verifier_crashed")
        elif "statistics_agree" in low or "recomputation" in low:
            labels.append("recomputation_disagreement")
        elif "constraint" in low or "feasibility" in low:
            labels.append("constraint_violation")
        elif "residual" in low or "conflict" in low:
            labels.append("residual_conflicts")
        elif "structure_matches_template" in low:
            labels.append("template_structure")
        elif "covers_every_plan" in low:
            labels.append("per_plan_coverage")
        elif "referential" in low:
            labels.append("referential_integrity")
        elif "outputs_present" in low:
            labels.append("missing_outputs")
        elif "execution_was_real" in low:
            labels.append("solver_crash")
        else:
            labels.append("other")
    return sorted(set(labels))


async def run_one(index: int, sem: asyncio.Semaphore, prefix: str = "r") -> dict:
    run_id = f"{prefix}{index:02d}"
    record: dict = {"run_id": run_id, "model": MODEL, "started_at": time.time()}

    async with sem:
        started = time.time()
        try:
            autopilot = CUMCMAutopilot(workspace=str(WORKSPACE))
            result = await autopilot.run(
                problem_files=FILES,
                competition="CUMCM",
                run_id=run_id,
            )
            record.update({
                "status": result.status,
                "verification": result.verification,
                "paper_pdf": bool(result.paper_pdf),
                "output_dir": result.output_dir,
                "blockers": result.blockers,
                "blocker_categories": _categorise(result.blockers),
                "stages": {k: v["status"] for k, v in result.stage_summary.items()},
                "stage_details": {k: v["detail"][:160] for k, v in result.stage_summary.items()},
                "notes": result.notes,
            })
        except Exception as exc:
            record.update({
                "status": "crashed",
                "verification": "",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
                "blockers": [],
                "blocker_categories": ["harness_exception"],
            })
        record["wall_seconds"] = round(time.time() - started, 1)
        record["converged"] = (
            record.get("status") == "completed" and record.get("verification") == "PASS"
        )

        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / f"{run_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with RECORDS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        flag = "CONVERGED" if record["converged"] else "NOT CONVERGED"
        print(
            f"[{run_id}] {flag} status={record.get('status')} "
            f"verification={record.get('verification') or '-'} "
            f"{record['wall_seconds']}s "
            f"{record.get('blocker_categories') or ''}",
            flush=True,
        )
        return record


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--prefix", default="r")
    args = parser.parse_args()

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not args.append and RECORDS.exists():
        RECORDS.unlink()

    print(f"model={MODEL} base_url={os.environ['OPENAI_BASE_URL']}", flush=True)
    print(f"runs={args.runs} parallel={args.parallel} workspace={WORKSPACE}", flush=True)

    sem = asyncio.Semaphore(args.parallel)
    t0 = time.time()
    records = await asyncio.gather(
        *(run_one(i, sem, args.prefix) for i in range(1, args.runs + 1))
    )

    ok = sum(1 for r in records if r.get("converged"))
    print()
    print("=" * 68)
    print(f"CONVERGED {ok}/{len(records)}  "
          f"({100.0 * ok / len(records):.1f}%)  "
          f"wall={round(time.time() - t0, 1)}s")
    for r in records:
        print(f"  {r['run_id']}: {'PASS' if r.get('converged') else 'FAIL':<4} "
              f"{r.get('status'):<10} {r.get('wall_seconds'):>7}s "
              f"{','.join(r.get('blocker_categories') or [])}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
