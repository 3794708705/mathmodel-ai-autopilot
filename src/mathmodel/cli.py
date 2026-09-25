"""MathModel AI — command line entry point.

The user-facing surface is deliberately small: upload the competition files
and start. Everything else is reported as progress, not as configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mathmodel.autopilot import CUMCMAutopilot, RunState


def _ensure_utf8_console() -> None:
    """Chinese output must not crash on a legacy Windows code page."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _print_result(result, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    print("")
    print("=" * 68)
    print(f"CUMCM Autopilot — run {result.run_id}")
    print("=" * 68)
    print(f"状态        : {result.status}")
    print(f"验证        : {result.verification or '(未运行)'}")
    print(f"运行目录    : {result.run_dir}")
    if result.paper_pdf:
        print(f"论文 PDF    : {result.paper_pdf}")
    if result.output_dir:
        print(f"交付目录    : {result.output_dir}")
    print("")
    print("阶段:")
    for name, info in result.stage_summary.items():
        detail = info.get("detail", "")
        print(f"  [{info['status']:<9}] {name:<14} {detail[:90]}")
    if result.pending_questions:
        print("")
        print("需要你确认:")
        for question in result.pending_questions:
            print(f"  - ({question.question_id}) {question.question}")
    if result.blockers:
        print("")
        print("阻塞:")
        for blocker in result.blockers:
            print(f"  - {blocker}")
    if result.notes:
        print("")
        print("备注:")
        for note in result.notes:
            print(f"  - {note}")


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_console()
    parser = argparse.ArgumentParser(
        prog="mathmodel",
        description="CUMCM Autopilot — upload the competition files and run the full workflow.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="start a new run from uploaded files")
    run_parser.add_argument(
        "--problem", "-p", nargs="+", required=True,
        help="problem statement file(s) and attachments",
    )
    run_parser.add_argument("--workspace", default="runs", help="where run directories live")
    run_parser.add_argument("--run-id", default=None, help="explicit run id")
    run_parser.add_argument("--sandbox-image", default="mathmodel-ai-autopilot:latest")
    run_parser.add_argument("--paper-image", default="mathmodel-ai-paper:phase6")
    run_parser.add_argument("--json", action="store_true", help="machine-readable output")

    resume_parser = sub.add_parser("resume", help="continue a paused or interrupted run")
    resume_parser.add_argument("run_dir")
    resume_parser.add_argument(
        "--answer", "-a", action="append", default=[],
        help="answer in the form QUESTION_ID=text",
    )
    resume_parser.add_argument("--sandbox-image", default="mathmodel-ai-autopilot:latest")
    resume_parser.add_argument("--paper-image", default="mathmodel-ai-paper:phase6")
    resume_parser.add_argument("--json", action="store_true")

    status_parser = sub.add_parser("status", help="show the persisted state of a run")
    status_parser.add_argument("run_dir")
    status_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "status":
        state = RunState.load(args.run_dir)
        summary = state.summary()
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"run      : {summary['run_id']}")
            print(f"status   : {summary['status']}")
            print(f"stage    : {summary['current_stage']}")
            for name, info in summary["stages"].items():
                print(f"  [{info['status']:<9}] {name:<14} {info['detail'][:90]}")
            for question in summary["pending_questions"]:
                print(f"  ? ({question['question_id']}) {question['question']}")
        return 0

    if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print(
            "错误: 未检测到模型凭据。请设置 DEEPSEEK_API_KEY（或 OPENAI_API_KEY）。",
            file=sys.stderr,
        )
        return 2

    autopilot = CUMCMAutopilot(
        workspace=args.workspace if args.command == "run" else Path(args.run_dir).parent,
        sandbox_image=args.sandbox_image,
        paper_image=args.paper_image,
    )

    if args.command == "run":
        result = asyncio.run(
            autopilot.run(problem_files=[Path(p) for p in args.problem], run_id=args.run_id)
        )
    else:
        answers = {}
        for item in args.answer:
            if "=" in item:
                key, value = item.split("=", 1)
                answers[key.strip()] = value.strip()
        result = asyncio.run(autopilot.resume(args.run_dir, answers=answers or None))

    _print_result(result, as_json=args.json)
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
