"""Run the CUMCM Autopilot end to end on the real 2026 CUMCM D-problem case.

Uses whichever OpenAI-compatible provider is reachable: the DeepSeek API when
it has balance, otherwise the local MathModel gateway on 127.0.0.1:7863.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _user_env(name: str) -> str:
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"[System.Environment]::GetEnvironmentVariable('{name}','User')"],
        capture_output=True, text=True,
    ).stdout.strip()


MODE = os.environ.get("MATHMODEL_PROVIDER", "local")

if MODE == "deepseek":
    os.environ.setdefault("DEEPSEEK_API_KEY", _user_env("DEEPSEEK_API_KEY"))
    os.environ["DEFAULT_PROVIDER"] = "deepseek"
else:
    os.environ["OPENAI_API_KEY"] = _user_env("MATHMODEL_LOCAL_7863_API_KEY")
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:7863/v1"
    os.environ["OPENAI_DEFAULT_MODEL"] = os.environ.get(
        "MATHMODEL_LOCAL_MODEL", "global:deepseek-v4.1-flash"
    )
    os.environ["DEFAULT_PROVIDER"] = "openai"

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


async def main() -> int:
    settings = get_settings()
    provider = settings.default_provider
    print(f"provider={provider} model={settings.openai_default_model} "
          f"base_url={os.environ.get('OPENAI_BASE_URL', '-')}", flush=True)

    autopilot = CUMCMAutopilot(workspace=str(ROOT / "runs"))
    result = await autopilot.run(
        problem_files=FILES,
        competition="CUMCM",
        run_id=os.environ.get("MATHMODEL_RUN_ID", "cumcm-2026-dti-e2e"),
    )

    (ROOT / ".tmp" / "last_result.json").write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("STATUS:", result.status, flush=True)
    print("VERIFICATION:", result.verification, flush=True)
    print("PDF:", result.paper_pdf, flush=True)
    print("OUTPUT:", result.output_dir, flush=True)
    for name, info in result.stage_summary.items():
        print(f"  [{info['status']:<9}] {name:<14} {info['detail'][:110]}", flush=True)
    for blocker in result.blockers:
        print("BLOCKER:", blocker[:400], flush=True)
    for note in result.notes:
        print("NOTE:", note[:300], flush=True)
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
