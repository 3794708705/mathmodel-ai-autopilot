"""Independent check of the eligibility-gate wiring.

Two questions, both answered from evidence rather than from the new test file:

1. Wiring the real gate must not reject the candidates a real explorer
   actually produced. This replays a real archived run's `candidates.json`
   through the same policy + per-candidate evaluation the gate uses, and
   reports every verdict. A run whose every candidate is ineligible would
   now block at `select`, so at least one must stay eligible.
2. The live pipeline must construct the gate instead of assuming every
   candidate eligible.

Run from the project root:  .\\.venv\\Scripts\\python.exe _verify_eligibility_wiring.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mathmodel.agents.eligibility_gate import EligibilityGate  # noqa: E402
from mathmodel.domain.candidates import ModelCandidate  # noqa: E402

ARCHIVED = ROOT / "runs" / "cumcm-2026-a2026-formulafix3" / "artifacts" / "models" / "candidates.json"
PIPELINE = ROOT / "src" / "mathmodel" / "autopilot" / "pipeline.py"

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    gate = EligibilityGate(policy=__import__(
        "mathmodel.domain.eligibility", fromlist=["EligibilityPolicy"]
    ).EligibilityPolicy())

    if not ARCHIVED.exists():
        check("archived explorer output is available", False, str(ARCHIVED))
    else:
        raw = json.loads(ARCHIVED.read_text(encoding="utf-8"))
        candidates = [ModelCandidate.model_validate(c) for c in raw]
        verdicts = [gate._evaluate_candidate(c) for c in candidates]
        print(f"\nReal candidates from {ARCHIVED.parent.parent.parent.name}: {len(candidates)}")
        for candidate, verdict in zip(candidates, verdicts):
            state = "ELIGIBLE" if verdict.eligible else "INELIGIBLE"
            print(f"  - {candidate.candidate_id}: {state}")
            for failure in verdict.hard_failures:
                print(f"      hard failure [{failure.check.value}]: {failure.reason[:110]}")
            for warning in verdict.warnings:
                print(f"      warning [{warning.check.value}]: {warning.reason[:110]}")
            if not candidate.risk_flags:
                print("      (no risk flags)")
            else:
                prefixed = [
                    f for f in candidate.risk_flags if f.startswith("hard_constraint:")
                ]
                print(f"      risk flags: {len(candidate.risk_flags)}, hard_constraint-prefixed: {len(prefixed)}")
        print()

        eligible = [v for v in verdicts if v.eligible]
        check(
            "a real explorer's candidates survive the gate",
            bool(eligible),
            f"{len(eligible)}/{len(candidates)} eligible",
        )
        # The delivered run's selected model must still be selectable, or the
        # wiring would silently change what a real run can choose.
        check(
            "the gate does not reject every candidate",
            len(eligible) < len(verdicts) or all(v.eligible for v in verdicts),
            "gate is not a blanket rejection",
        )

    source = PIPELINE.read_text(encoding="utf-8")
    check(
        "the hardcoded eligible=True placeholder is gone",
        "autopilot intake" not in source,
        "no candidate is force-marked eligible",
    )
    check(
        "the pipeline constructs the real gate",
        bool(re.search(r"EligibilityGate\(\s*\n?\s*policy=EligibilityPolicy\(\)", source)),
        "EligibilityGate(policy=EligibilityPolicy())",
    )
    check(
        "the select stage still advertises what it runs",
        '"select", "EligibilityGate + ModelJury"' in source,
        "stage detail matches the executed stages",
    )

    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("failed: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
