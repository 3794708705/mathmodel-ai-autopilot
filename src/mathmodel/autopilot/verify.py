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
import math
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

        The checks that already PASS are named at the end. Repair feedback that
        lists only failures tells the solver nothing about what to leave alone,
        and a repair that rewrites working code is a regression the solver never
        sees coming. Measured on this project: a solver whose own scheme
        validation was sound in every stated measurement was still rewritten
        five times, and repeated failure classes recurred because each repair
        re-touched code that had already been correct.
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
        passed_names = [
            check.name
            for check in self.checks
            if check.status == CheckStatus.PASS
        ]
        if passed_names and lines:
            lines.append(
                "[context] these checks already PASS and your repair must keep "
                "them passing: " + ", ".join(passed_names[:40])
            )
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


def read_output_sheet_names(path: Path) -> list[str]:
    """Worksheet names of an output artifact, in workbook order."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            return list(wb.sheetnames)
        finally:
            wb.close()
    return []


def describe_solution_files(
    output_dir: str | Path, required_outputs: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """The authoritative shape of each produced output file.

    A generated verifier that has to infer this layout guesses wrong, and each
    guess turns into a false accusation against the solver: it filters
    worksheets by name and finds nothing on a template whose sheet is only
    `Sheet1`, or it folds the time column into the field values so that every
    sheet spans 0..t_end and no value range matches. Handing it the real shape
    removes the guesswork.

    The first and last data rows are included as well, so the verifier can
    identify a sheet by what is actually in it. Told only the sheet names it
    picked the wrong one and reported moisture 28.0 where the correct initial
    moisture was 2.55 — it had read the temperature sheet, because the
    temperature sheet comes first.
    """
    output_dir = Path(output_dir)
    described: dict[str, list[dict[str, Any]]] = {}
    for name in required_outputs:
        path = output_dir / name
        if not path.exists():
            continue
        sheets: list[dict[str, Any]] = []
        try:
            import openpyxl

            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            try:
                for ws in wb.worksheets:
                    rows = [list(r) for r in ws.iter_rows(values_only=True)]
                    rows = [
                        r
                        for r in rows
                        if any(v is not None and str(v).strip() != "" for v in r)
                    ]
                    if not rows:
                        sheets.append(
                            {"sheet": ws.title, "data_rows": 0, "columns": 0}
                        )
                        continue
                    headers = ["" if h is None else str(h).strip() for h in rows[0]]
                    sheets.append({
                        "sheet": ws.title,
                        "data_rows": len(rows) - 1,
                        "columns": len(headers),
                        "header_row": headers[:26],
                        "first_data_row": _row_sample(rows[1]) if len(rows) > 1 else [],
                        "last_data_row": _row_sample(rows[-1]) if len(rows) > 1 else [],
                    })
            finally:
                wb.close()
        except Exception:
            continue
        if sheets:
            described[name] = sheets
    return described


def _row_sample(row: list[Any], limit: int = 26) -> list[Any]:
    """One row of a solution sheet, rounded so it can be read at a glance."""
    sample: list[Any] = []
    for value in row[:limit]:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            sample.append(round(float(value), 6))
        else:
            sample.append(str(value))
    return sample


# ═══════════════════════════════════════════════════════════════
# Template header semantics
# ═══════════════════════════════════════════════════════════════

# A template may abbreviate the interior of a numeric axis with an ellipsis,
# e.g. `0, 0.1, 0.2, …, 2`. That cell is a placeholder, not a heading, so it is
# never compared literally. It is accepted only when the columns standing in
# for it are an unambiguous arithmetic continuation of the numbers printed
# before it; when the progression cannot be derived the template still fails,
# because guessing the intended axis would accept any number of columns.
ELLIPSIS_CELLS = {"…", "...", "...."}

# `Sheet1` is the spreadsheet default, not a heading the answer has to
# reproduce, so a template sheet carrying such a name is not required.
GENERIC_SHEET_RE = re.compile(r"^sheet\s*\d*$", re.IGNORECASE)

# The problem states how often a table must be sampled: "...每隔 60 s ... 保存到
# 文件 result3.xlsx". The interval governing a file is the last one stated
# before that file's name.
TIME_STEP_RE = re.compile(r"每隔\s*([\d.]+)\s*(?:s|秒)", re.IGNORECASE)
RESULT_FILE_RE = re.compile(r"(result\d+\.xlsx)", re.IGNORECASE)

# A final-state threshold on moisture, e.g. "药材各处的水分浓度应低于 0.15 kg/kg".
MOISTURE_LIMIT_RE = re.compile(r"(低于|小于|不超过|高于|大于)\s*([\d.]+)\s*kg/kg")

# ...and the phrase marking the outputs that report the END of drying, which is
# what such a threshold is the acceptance criterion for. The statement wraps
# mid-phrase in the PDF ("烘干所需\n要的时间"), so whitespace is allowed between
# the characters. A bare "烘干结束时间" is deliberately NOT matched: that is a
# row label inside the answer tables, and it sits nearer the wrong result file.
DRYING_TIME_RE = re.compile(
    r"烘干\s*(?:所\s*需\s*(?:要\s*)?(?:的\s*)?时\s*间|时\s*长)"
)


def _as_number(cell: Any) -> Optional[float]:
    """The numeric value of a cell, or None when it is not a number."""
    if isinstance(cell, bool):
        return None
    if isinstance(cell, (int, float)):
        return float(cell)
    if isinstance(cell, str):
        try:
            return float(cell.strip())
        except ValueError:
            return None
    return None


def is_ellipsis(cell: Any) -> bool:
    return isinstance(cell, str) and cell.strip() in ELLIPSIS_CELLS


def _cells_match(expected: Any, actual: Any) -> bool:
    """Compare two header cells, numerically when both hold numbers."""
    want, got = _as_number(expected), _as_number(actual)
    if want is not None and got is not None:
        return abs(want - got) < 1e-9
    return str(expected).strip().lower() == str(actual).strip().lower()


def _literal_header_mismatches(expected: list[str], actual: list[str]) -> list[str]:
    expected_norm = [h.strip().lower() for h in expected if h.strip()]
    actual_norm = [h.strip().lower() for h in actual]
    if len(actual_norm) < len(expected_norm):
        return [f"expected >= {len(expected_norm)} columns, got {len(actual_norm)}"]
    return [
        f"column {index + 1}: expected {expected_norm[index]!r}, "
        f"got {actual_norm[index]!r}"
        for index in range(len(expected_norm))
        if actual_norm[index] != expected_norm[index]
    ]


def _ellipsis_header_mismatches(
    expected: list[str], actual: list[str], position: int
) -> list[str]:
    prefix = [h.strip() for h in expected[:position]]
    suffix = [h.strip() for h in expected[position + 1:]]
    produced = [str(h).strip() for h in actual]

    minimum = len(prefix) + len(suffix) + 1
    if len(produced) < minimum:
        return [
            f"expected at least {minimum} columns (the template's '…' stands for "
            f"at least one omitted column), got {len(produced)}"
        ]
    for index, want in enumerate(prefix):
        if not _cells_match(want, produced[index]):
            return [f"column {index + 1}: expected {want!r}, got {produced[index]!r}"]
    for offset, want in enumerate(suffix):
        index = len(produced) - len(suffix) + offset
        if not _cells_match(want, produced[index]):
            return [f"column {index + 1}: expected {want!r}, got {produced[index]!r}"]

    omitted = produced[len(prefix): len(produced) - len(suffix)]

    known = [v for v in (_as_number(h) for h in prefix) if v is not None]
    if len(known) < 2:
        return [
            "the template's '…' cannot be expanded: fewer than two numbers are "
            "printed before it, so the omitted step is unknown"
        ]
    step = known[-1] - known[-2]
    if abs(step) < 1e-12:
        return ["the template's '…' cannot be expanded: the printed step is zero"]
    if any(abs((b - a) - step) > 1e-9 for a, b in zip(known, known[1:])):
        return [
            "the template's '…' cannot be expanded: the numbers printed before it "
            "are not an arithmetic progression"
        ]

    values = [_as_number(h) for h in omitted]
    unparsed = [h for h, v in zip(omitted, values) if v is None]
    if unparsed:
        return [
            f"the template's '…' stands for numbers continuing {known[-1]:g} by "
            f"{step:g}, but columns {unparsed[:4]!r} are not numbers"
        ]
    numbers = [v for v in values if v is not None]

    problems: list[str] = []
    if abs((numbers[0] - known[-1]) - step) > 1e-9:
        problems.append(
            f"column {len(prefix) + 1}: expected {known[-1] + step:g} "
            f"(continuing by {step:g}), got {numbers[0]:g}"
        )
    for index, (a, b) in enumerate(zip(numbers, numbers[1:])):
        if abs((b - a) - step) > 1e-9:
            problems.append(
                f"column {len(prefix) + index + 2}: expected {a + step:g} "
                f"(continuing by {step:g}), got {b:g}"
            )
    # When the template prints a number after the ellipsis, the progression has
    # to reach it; otherwise the omitted span is the wrong length.
    tail = _as_number(suffix[0]) if suffix else None
    if tail is not None and abs((numbers[-1] + step) - tail) > 1e-9:
        problems.append(
            f"the '…' must expand to values ending one step ({step:g}) before "
            f"{tail:g}, but the last column before it is {numbers[-1]:g}"
        )
    return problems[:4]


def _repeated_final_label(actual: list[str]) -> Optional[float]:
    """Return the value when a header ends with the same label twice.

    Measured on the real A problem: a generated solver wrote a radius header of
    23 columns ending `'1.8', '1.9', '2', '2'` -- the final label duplicated, so
    the header carried one column more than the template. The template's
    ellipsis check reported only that the omitted span did not end one step
    before the number after it, which describes the symptom; the FORM to name is
    "the last label is repeated, so you have one column too many".
    """
    values = [_as_number(header) for header in actual[-2:]]
    if len(values) != 2 or any(value is None for value in values):
        return None
    if abs(values[0] - values[1]) <= 1e-9:
        return values[-1]
    return None


def header_mismatches(expected: list[str], actual: list[str]) -> list[str]:
    """How an output header fails to match the supplied template header."""
    inner = _header_mismatches_inner(expected, actual)
    repeated = _repeated_final_label(actual)
    if inner and repeated is not None:
        return [
            f"the last column label is repeated: this header ends with "
            f"{repeated:g} twice, so it carries one column more than the axis "
            f"needs, and every label before the duplicate is shifted"
        ] + inner
    return inner


def _header_mismatches_inner(expected: list[str], actual: list[str]) -> list[str]:
    positions = [i for i, h in enumerate(expected) if is_ellipsis(h)]
    if not positions:
        return _literal_header_mismatches(expected, actual)
    if len(positions) > 1:
        return [
            f"the template header has {len(positions)} ellipsis columns; the "
            f"omitted axes cannot be derived"
        ]
    return _ellipsis_header_mismatches(expected, actual, positions[0])


def problem_process_duration(problem_text: str) -> Optional[tuple[float, float]]:
    """The duration range, in seconds, the statement gives for the process.

    Measured on this project: a solver dried the material from 2.55 to below
    0.15 kg/kg within about an hour, while the statement says the process lasts
    2-3 days. It therefore passed the stated final-moisture target VACUOUSLY --
    the target was met trivially because everything was dried almost at once --
    and nothing host-side noticed, because the only duration cross-check lived
    in a prompt the solver could ignore. This makes the statement's own duration
    available to the host so that a duration can be tested rather than advised.

    Returns `(low, high)` in seconds, or None when the statement gives no
    duration. A single stated duration returns it as both bounds.
    """
    if not problem_text:
        return None
    # Take the BEST candidate, not the first. The real 2026 A statement mentions
    # several durations ("table 3 lists only 3h", "problem 4 lists up to the end
    # time") before it says the drying lasts 2-3 days, and reading the first one
    # returned 4 hours. A duration that describes the PROCESS is the one near a
    # process word, and a range is more informative than a bare figure.
    best: Optional[tuple[int, re.Match[str]]] = None
    for match in _PROCESS_DURATION_RE.finditer(problem_text):
        score = 0
        if match.group("unit") is not None:
            score += 2  # an explicit range beats a lone number
        window = problem_text[max(0, match.start() - 30) : match.start()]
        if _PROCESS_WORD_RE.search(window):
            score += 4
        if score > 0 and (best is None or score > best[0]):
            best = (score, match)
    if best is None:
        return None
    match = best[1]
    if match.group("low") is not None:
        low = float(match.group("low"))
        high = float(match.group("high")) if match.group("high") else low
        unit = match.group("unit")
    else:
        low = high = float(match.group("low2"))
        unit = match.group("unit2")
    scale = _DURATION_UNITS.get(unit.strip().lower())
    if scale is None:
        return None
    if high < low:
        low, high = high, low
    return low * scale, high * scale


_PROCESS_DURATION_RE = re.compile(
    r"(?P<low>\d+(?:\.\d+)?)\s*(?:[-~～—至到]|\s*到\s*)\s*(?P<high>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>天|日|小时|h|hr|hour|hours|day|days)"
    r"|(?P<low2>\d+(?:\.\d+)?)\s*(?P<unit2>天|日|小时|h|hr|hour|hours|day|days)"
)

_PROCESS_WORD_RE = re.compile(
    r"持续|历时|干燥|烘干|干燥过程|process|lasts|lasting|duration|drying"
)

_DURATION_UNITS = {
    "天": 86400.0,
    "日": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
    "小时": 3600.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}


def problem_transport_ceiling(problem_text: str) -> Optional[float]:
    """The largest moisture-removal rate, in kg/s, the stated physics permits.

    A surface exchange cannot remove more moisture than `h_m * rho * (C - C_air)`
    over the exposed area, so a solution that loses mass faster than that ceiling
    is impossible regardless of how it discretises anything. Measured on this
    project: a generated solver removed 0.62 kg of moisture in 180 s -- about 66x
    this ceiling -- and every check the solver wrote for itself passed. This
    bound depends only on the problem's own constants, so unlike the solver's
    own conservation audit it cannot be talked out of.

    Note on units, because getting this wrong once cost a correct design: `h_m`
    has velocity units, and the concentration here is in kg/kg, so the flux needs
    the density to become kg/(m^2 s). Dropping `rho` understates the ceiling by
    that factor and makes the test reject physically-reasonable answers.

    Returns None when the statement does not give the constants, so a problem
    without them simply gets no such check.
    """
    limits = problem_transport_limits(problem_text)
    return None if limits is None else limits["ceiling_rate"]


def problem_transport_limits(problem_text: str) -> Optional[dict[str, float]]:
    """The stated coefficients and the ceiling they imply, for the gate to use.

    `ceiling_rate` is the largest moisture removal rate in kg/s;
    `initial_moisture_mass` is the moisture the specimen starts with, in kg. A
    check needs both, because the ceiling bounds the RATE while the initial mass
    converts a measured concentration drop into an actual rate.
    """
    if not problem_text:
        return None
    geometry = _GEOMETRY_RE.search(problem_text)
    coefficient = _MASS_COEFFICIENT_RE.search(problem_text)
    density = _DENSITY_RE.search(problem_text)
    moisture = _INITIAL_MOISTURE_RE.search(problem_text)
    if not (geometry and coefficient and density and moisture):
        return None
    length = float(geometry.group(1)) * _LENGTH_UNITS.get(geometry.group(2), 0.0)
    radius = float(geometry.group(3)) * _LENGTH_UNITS.get(geometry.group(4), 0.0)
    coefficient_value = float(coefficient.group(1)) * 10.0 ** (-int(coefficient.group(2)))
    density_value = float(density.group(1) or density.group(2))
    moisture_value = float(moisture.group(1))
    if min(length, radius, coefficient_value, density_value, moisture_value) <= 0.0:
        return None
    area = 2.0 * math.pi * radius * length
    volume = math.pi * radius * radius * length
    dry_mass = density_value * volume
    return {
        "ceiling_rate": coefficient_value * density_value * moisture_value * area,
        "initial_moisture_mass": moisture_value * dry_mass,
        "initial_moisture": moisture_value,
        "radius": radius,
    }


_GEOMETRY_RE = re.compile(
    r"长\s*为\s*([\d.]+)\s*(cm|mm|m|厘米|毫米|米)[^。]{0,20}?"
    r"半径\s*为\s*([\d.]+)\s*(cm|mm|m|厘米|毫米|米)"
)
_MASS_COEFFICIENT_RE = re.compile(
    r"传质系数\s*为\s*([\d.]+)\s*[×xX*]\s*10\s*[-−–—]\s*(\d+)\s*m/s"
)
_DENSITY_RE = re.compile(r"密度\s*为\s*([\d.]+)\s*kg/m\s*\^?3?|密度\s*为\s*([\d.]+)")
_INITIAL_MOISTURE_RE = re.compile(r"含水率[^0-9]{0,24}([\d.]+)\s*kg/kg")
_LENGTH_UNITS = {
    "m": 1.0,
    "米": 1.0,
    "cm": 1e-2,
    "厘米": 1e-2,
    "mm": 1e-3,
    "毫米": 1e-3,
}


def problem_time_steps(problem_text: str) -> dict[str, float]:
    """Sampling interval, in seconds, the problem demands for each output file.

    Only files the statement gives an explicit interval for are returned, so a
    problem that states no sampling rate simply gets no such check.
    """
    if not problem_text:
        return {}
    mentions = list(RESULT_FILE_RE.finditer(problem_text))
    steps: dict[str, float] = {}
    previous_end = 0
    for mention in mentions:
        name = mention.group(1)
        if name in steps:
            previous_end = mention.end()
            continue
        window = problem_text[previous_end: mention.start()]
        found = list(TIME_STEP_RE.finditer(window))
        if found:
            steps[name] = float(found[-1].group(1))
        previous_end = mention.end()
    return steps


def problem_final_state_targets(
    problem_text: str, required_outputs: list[str]
) -> list[tuple[str, str, float]]:
    """Final-state thresholds the problem states for a named output.

    The statement says "药材各处的水分浓度应低于 0.15 kg/kg" and then asks for the
    drying time; that threshold is the acceptance criterion for the outputs that
    report the end of drying, and only those.
    """
    if not problem_text:
        return []
    limit = MOISTURE_LIMIT_RE.search(problem_text)
    if not limit:
        return []
    comparator = "<" if limit.group(1) in ("低于", "小于", "不超过") else ">"
    value = float(limit.group(2))

    targets: list[tuple[str, str, float]] = []
    previous_end = 0
    for mention in RESULT_FILE_RE.finditer(problem_text):
        name = mention.group(1)
        window = problem_text[previous_end: mention.start()]
        previous_end = mention.end()
        if name not in required_outputs or any(t[0] == name for t in targets):
            continue
        if DRYING_TIME_RE.search(window):
            targets.append((name, comparator, value))
    return targets


# Markers that a model's equation really is a differential equation the solver
# has to integrate, as opposed to an algebraic definition. An algebraic
# assignment has no derivative operator, so it needs no scheme validation.
_DERIVATIVE_MARKERS = (
    "∂", "\\partial", "d/dt", "d/dr", "d/dx", "d²", "d^2",
)
_DIFFERENTIAL_FAMILY_MARKERS = (
    "pde", "ode", "differential", "parabolic", "hyperbolic", "elliptic", "微分方程",
)


def model_integrates_differential_equations(model: Any) -> bool:
    """Whether the model states a differential equation the solver must integrate.

    This decides two things downstream: whether the generated program has to
    validate its numerical scheme, and which cause-level hint the repair loop
    gives when an independent recomputation disagrees. A combinatorial model
    (assignment equations only) answers False and is unaffected.
    """
    for equation in getattr(model, "equations", None) or []:
        text = " ".join(
            str(getattr(equation, field, "") or "")
            for field in ("expression", "description", "name")
        )
        if any(marker in text for marker in _DERIVATIVE_MARKERS):
            return True
    family = str(getattr(model, "model_family", "") or "").lower()
    return any(marker in family for marker in _DIFFERENTIAL_FAMILY_MARKERS)


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


def _scheme_self_check(reported: Any, *, required: bool) -> VerificationCheck:
    """Validate the solver's own record of validating its numerical scheme.

    A scheme can drop a control-volume factor or rebuild its time-derivative
    right-hand side from the current iterate, still exit 0 and report plausible
    numbers. The solver is required to compare its scheme against a case with a
    known answer and report the measured deviation; this check reads that record.

    A missing record fails only when the model actually integrates a
    differential equation, so a combinatorial model is unaffected.
    """
    if not isinstance(reported, dict) or not reported:
        if not required:
            return VerificationCheck(
                name="scheme_self_check_passed",
                category="correctness",
                status=CheckStatus.NOT_APPLICABLE,
                detail=(
                    "The model states no differential equation, so no numerical "
                    "scheme needed validating"
                ),
            )
        return VerificationCheck(
            name="scheme_self_check_passed",
            category="correctness",
            status=CheckStatus.FAIL,
            detail=(
                "The model integrates a differential equation, but the program "
                "reported no 'self_check' object, so its numerical scheme was never "
                "validated against a case with a known answer"
            ),
            evidence={"reported_self_check": reported},
        )

    passed = reported.get("passed")
    case = str(reported.get("case", "") or "").strip()
    error = _as_number(reported.get("max_abs_error"))
    tolerance = _as_number(reported.get("tolerance"))
    initial_error = _as_number(reported.get("initial_profile_error"))
    steady_error = _as_number(reported.get("steady_state_error"))
    profile_span = _as_number(reported.get("profile_span"))
    reference_span = _as_number(reported.get("reference_span"))
    # The VALUE of a flat profile is what separates an over-fast conductance from
    # a boundary condition that never acts, so the contract asks for it.
    profile_min = _as_number(reported.get("profile_min"))
    profile_max = _as_number(reported.get("profile_max"))
    initial_value = _as_number(reported.get("initial_value"))
    boundary_value = _as_number(reported.get("boundary_value"))

    issues: list[str] = []
    if not isinstance(passed, bool):
        issues.append("'passed' is not a boolean")
    elif not passed:
        # A `passed=false` verdict must be ACCOUNTABLE to the program's own
        # numbers. Measured in practice: a solver reported max_abs_error=0.0148
        # against tolerance=1.0, initial_profile_error=0.283, steady_state_error=
        # 4.3e-11 and profile_span 14.149 against reference_span 14.173 -- a
        # validated scheme by its own evidence -- yet set passed=false, and the
        # bare boolean blocked the run. Reading the boolean in place of the
        # evidence inverts this system's rule that a solver's claims are never
        # taken on trust, so name the contradiction instead.
        failed_criteria = reported.get("failed_criteria")
        if isinstance(failed_criteria, str):
            failed_criteria = [failed_criteria]
        named_failures = (
            [str(item).strip() for item in failed_criteria if str(item).strip()]
            if isinstance(failed_criteria, list)
            else []
        )
        # Deduplicate while keeping the program's own order: a solver that lists
        # the same criterion twice should not read as two separate faults.
        named_failures = list(dict.fromkeys(named_failures))
        # Whether the order study is the ONLY thing the program reported failing.
        # A negative order is evidence about the TEST, and if nothing else failed
        # the scheme is not implicated at all -- see the negative-order branch.
        other_criteria_failed = [
            name for name in named_failures if name != "manufactured_order"
        ]
        measurements_within_tolerance = (
            error is not None and tolerance is not None and error <= tolerance
        )
        if named_failures:
            issues.append(
                "the program named these criteria as failed: "
                + ", ".join(named_failures[:6])
            )
        elif measurements_within_tolerance:
            issues.append(
                "the program's verdict contradicts its own measurements: it "
                f"reported passed=false while max_abs_error={error:g} is inside the "
                f"tolerance {tolerance:g} that it stated. Name the criterion that "
                "actually failed in `failed_criteria`, or correct the verdict to "
                "match the measurements."
            )
        else:
            issues.append("the program itself reported passed=false")
    if error is None:
        issues.append("'max_abs_error' is missing or not numeric")
    if tolerance is None:
        issues.append("'tolerance' is missing or not numeric")
    # A tolerance the solver chooses for itself is only meaningful if it is small
    # compared with the quantity it is measuring. Measured on this project, two
    # solvers on the same problem declared tolerances of 1.0 and 0.005 -- a factor
    # of 200 -- so the strictness of this whole check was set by whichever number
    # the generation happened to pick, and a lax choice passes it trivially while a
    # strict one fails. The contract already requires `reference_span` (how much
    # the reference quantity varies), which is exactly the scale a tolerance has to
    # be small against.
    # A scheme that never checks its own conservation cannot detect the one
    # defect class that looks locally perfect: a spurious or missing factor in a
    # single term of a balance. Measured on this project, a generated solver put
    # `rho ~ 820` into its moisture surface flux but not into its moisture
    # storage, so surface exchange was ~820x too strong and the material dried
    # from 2.55 to below 0.15 kg/kg in under 180 s -- while its own scheme
    # validation passed every criterion it had written. A mass-balance residual
    # is what makes that visible, and it is not something the solver can talk
    # its way out of, because it compares two of the solver's own numbers.
    balance_raw = reported.get("mass_balance_relative_error")
    energy_balance_raw = reported.get("energy_balance_relative_error")
    if balance_raw is None and energy_balance_raw is None:
        issues.append(
            "your program does not report a conservation audit. Report "
            "`mass_balance_relative_error` (and `energy_balance_relative_error` "
            "for the thermal field) in `self_check`: the difference between the "
            "change in stored quantity and the integral of the boundary flux over "
            "the whole simulation, divided by their scale. Without it, a factor "
            "error in one term of a balance -- the single defect class that leaves "
            "every other check passing -- cannot be detected by you or by anyone "
            "reading your results"
        )
    else:
        for label, raw in (
            ("mass_balance_relative_error", balance_raw),
            ("energy_balance_relative_error", energy_balance_raw),
        ):
            if raw is None:
                continue
            try:
                residual = float(raw)
            except (TypeError, ValueError):
                issues.append(f"your reported `{label}` is not a number: {raw!r}")
                continue
            if not math.isfinite(residual):
                issues.append(f"your reported `{label}` is not finite: {residual!r}")
                continue
            if residual > CONSERVATION_RELATIVE_TOLERANCE:
                issues.append(
                    f"your own conservation audit reports `{label}` = "
                    f"{residual:.6g}, which is not a balanced scheme. A relative "
                    f"imbalance that large means the conserved quantity does not "
                    f"add up: check that EVERY term of that balance carries the "
                    f"same factors -- if your storage term contains the density, "
                    f"the flux terms must too, and if it does not, they must not "
                    f"either. Then check the same for any control-volume width "
                    f"factor"
                )

    if (
        tolerance is not None
        and reference_span is not None
        and reference_span > 0.0
        and tolerance > 0.05 * reference_span
    ):
        issues.append(
            f"your declared tolerance {tolerance:g} is not a meaningful test: your "
            f"reference quantity only spans {reference_span:g}, so a tolerance of "
            f"{tolerance:g} is {100.0 * tolerance / reference_span:.0f}% of the "
            f"whole range and almost any field would satisfy it. A tolerance has to "
            f"be small compared with what it measures. Choose one that is a small "
            f"fraction of the reference span (a few percent at most) and say why "
            f"that is the right scale"
        )

    if error is not None and tolerance is not None and error > tolerance:
        if error >= DIVERGED_ABSOLUTE or (
            reference_span is not None
            and reference_span > 0.0
            and error > 1e6 * reference_span
        ):
            # Not an accuracy problem: the validation run BLEW UP. Saying "the
            # deviation exceeds the tolerance" for an error of 1e108 tells the
            # solver nothing about what to look at. Measured on this project: a
            # solver whose main problem produced a physically sensible drying
            # time (43.5 h, corroborated independently) still reported
            # max_abs_error=7.18062e+108, because its validation case imposed a
            # DIRICHLET surface while the real problem uses CONVECTION -- so the
            # case exercised a boundary treatment the main run never used, and
            # that treatment is unstable.
            issues.append(
                f"your scheme DIVERGED in the validation case: the deviation is "
                f"{error:g}, which is not an accuracy shortfall but a blow-up "
                f"(reference span {reference_span:g}). Look at what your "
                f"validation case does DIFFERENTLY from the problem you are "
                f"actually solving -- in particular its boundary condition. A "
                f"validation case that imposes the surface value directly while "
                f"the real problem exchanges with the surroundings through a "
                f"conductance exercises a different boundary row, so an unstable "
                f"or inconsistently coupled surface in one of them will blow up "
                f"there while the main run looks fine. Check that the surface "
                f"value or flux is imposed through the same control-volume "
                f"balance as the interior, that no diagonal entry can vanish or "
                f"change sign, and that the case you chose is stable at the time "
                f"step you used"
            )
        else:
            issues.append(
                f"the measured deviation {error:g} exceeds the stated tolerance "
                f"{tolerance:g}"
            )
    if not case:
        issues.append("'case' does not say what the scheme was compared against")

    # A reference that disagrees with the scheme at t=0 is itself broken: the
    # A manufactured-solution convergence test measures the SCHEME rather than a
    # reference, so it survives the failure mode that keeps recurring here: an
    # analytic reference that is itself broken (a missing control-volume factor, a
    # wrong eigenvalue condition) and then blocks a scheme that was sound. A
    # scheme that has lost a geometric factor, or counts a flux twice, does not
    # converge at the order it claims.
    manufactured_order = _as_number(reported.get("manufactured_order"))
    expected_order = _as_number(reported.get("expected_order"))
    # A reported order is a CONCLUSION. The levels behind it are the evidence,
    # and a conclusion that its own evidence contradicts is not a finding about
    # the scheme. Measured on this project: a solver reported an order of
    # -0.0007 while its scheme produced output that satisfied every host-side
    # acceptance check, which is the signature of a wrong manufactured source
    # term (a saturated error gives order ~ 0 at any refinement) rather than a
    # broken scheme -- so the repair has to be aimed at the test.
    levels_auditable = False
    levels = reported.get("manufactured_levels")
    if isinstance(levels, list) and len(levels) >= 2:
        pairs: list[tuple[float, float]] = []
        for item in levels:
            if not isinstance(item, dict):
                continue
            n = _as_number(item.get("cells", item.get("n")))
            err = _as_number(item.get("error"))
            if n is not None and err is not None and n > 0 and err > 0:
                pairs.append((n, err))
        pairs.sort()
        if len(pairs) >= 2:
            levels_auditable = True
        if len(pairs) >= 2 and manufactured_order is not None:
            (n0, e0), (n1, e1) = pairs[0], pairs[-1]
            implied = math.log(e0 / e1) / math.log(n1 / n0)
            if implied > 0.5 * max(expected_order or 1.0, 1.0) and (
                manufactured_order <= 0.5 * (expected_order or 1.0)
            ):
                issues.append(
                    f"your manufactured-levels evidence contradicts the order you "
                    f"reported: the errors you listed fall from {e0:g} at {n0:g} "
                    f"cells to {e1:g} at {n1:g} cells, an implied order of "
                    f"{implied:.3g}, yet you reported manufactured_order="
                    f"{manufactured_order:g}. Your own refinement study CONVERGES, "
                    f"so the scheme is not the fault -- you miscomputed the order "
                    f"from your own data. Fix the order calculation (and the "
                    f"`passed`/`failed_criteria` verdict that depends on it) rather "
                    f"than changing the scheme"
                )

    if (
        manufactured_order is not None
        and expected_order is not None
        and expected_order > 0
        and manufactured_order < 0.5 * expected_order
    ):
        if not levels_auditable:
            # The gate still fails, so no standard moves. What changes is the
            # ATTRIBUTION: without the levels behind it, a near-zero order cannot
            # be told apart from a wrong manufactured source term, and a wrong
            # source term is the more likely cause -- a saturated error gives
            # order ~0 at any refinement. Saying "the scheme does not converge"
            # would send the repair into a discretisation that may be sound.
            issues.append(
                f"your manufactured-solution test reports order "
                f"{manufactured_order:g} where your scheme claims {expected_order:g}, "
                f"but you did not report the grid levels behind that number "
                f"(`manufactured_levels`), so it is not auditable. An order near zero "
                f"is produced both by a scheme that does not converge AND by a wrong "
                f"source term in the test -- a saturated error gives order ~0 at any "
                f"refinement -- and the second is the more common cause. Report the "
                f"cells and error for each level you refined over, and fix the test "
                f"before you change the scheme: re-derive the source term by "
                f"substituting your trial field into the governing equation term by "
                f"term, and check the trial field satisfies your boundary conditions "
                f"at every level"
            )
        elif manufactured_order <= 0.0:
            # A NEGATIVE observed order is a stronger statement than a small one:
            # refining makes the error BIGGER. That is not a scheme that lost an
            # order, it is one whose comparison is not measuring convergence at
            # all, and the usual causes are on the test side rather than the
            # discretisation side.
            issues.append(
                f"your manufactured-solution test shows order "
                f"{manufactured_order:g} where your scheme claims {expected_order:g}, "
                f"and a NEGATIVE order means the error GROWS as you refine — that is "
                f"not a scheme with a poor constant, it is a test that is not "
                f"measuring convergence. Before changing the scheme, check the test "
                f"itself: the source term derived from your trial field (substitute "
                f"the trial field into the governing equation symbolically term by "
                f"term and keep every term, including the ones that vanish for a "
                f"simpler field), whether the trial field satisfies your boundary "
                f"conditions at every grid level, whether you refine time and space "
                f"consistently (halving both, or holding one fixed and saying so), "
                f"and whether you compare at the SAME physical time at every level. "
                f"Measured on this project: a solver reported an order of -1.48674 "
                f"against a claimed 2"
            )
            if not other_criteria_failed:
                # The order study is the ONLY criterion that failed, and a
                # negative order is evidence about the TEST, not the scheme. Say
                # so plainly, because otherwise the solver will rebuild a scheme
                # that may already be right.
                #
                # Measured: runs/cumcm-2026-a2026-rebasefix attempt 3 solved the
                # problem correctly -- its drying passed the required threshold,
                # its own deviation from its reference was 0.0122, and the host
                # read valid output files -- and it was blocked by an order of
                # -1.95513 from a manufactured test whose source term cannot be
                # right. Do NOT treat this as evidence against the answer, and do
                # NOT rebuild the scheme: fix or replace the test.
                issues.append(
                    "IMPORTANT: the order study is the ONLY criterion that failed, "
                    "and a negative order is a statement about your TEST, not about "
                    "your scheme. Every other criterion passed, including the "
                    "deviation of your solution from its reference. So the "
                    "discretisation is not implicated: the likely fault is the "
                    "source term or the trial field of the manufactured case. "
                    "Repair the TEST -- or replace it with a simpler case whose "
                    "exact solution you can verify independently, such as a "
                    "constant-coefficient problem with a steady state -- and do not "
                    "rewrite the solver that produced your answer"
                )
        else:
            issues.append(
                f"your manufactured-solution convergence test shows order "
                f"{manufactured_order:g} where your scheme claims {expected_order:g}: "
                f"refining the grid is not reducing the error at the expected rate, "
                f"which means a structural fault in the discretisation (a lost "
                f"control-volume factor, a flux counted twice, a boundary conductance "
                f"measured on the wrong area) rather than a resolution problem. Do not "
                f"respond by refining the grid further"
            )

    # A reference that disagrees with the scheme at t=0 is itself broken: the
    # scheme starts from the initial condition it was handed, so that deviation
    # measures the REFERENCE, not the scheme. Saying so matters because the
    # generic wording ("the deviation exceeds the tolerance") sends the model to
    # re-derive a scheme that was right all along. Measured in practice: a
    # reference whose series coefficient had the wrong sign returned 90.4 at the
    # centre where the correct initial value was 28.0, and the resulting 64.96
    # "scheme error" blocked a run whose scheme reproduced T0 exactly.
    #
    # The t=0 disagreement must DOMINATE the reported deviation to count. Testing
    # it against the tolerance alone is wrong: a solver that states an unusually
    # tight tolerance reports a discretisation-level t=0 gap (1.6e-3 against a
    # 1e-3 tolerance) that says nothing about the reference, and the earlier
    # absolute test blamed the reference for it.
    reference_is_broken = (
        initial_error is not None
        and tolerance is not None
        and error is not None
        and initial_error > tolerance
        and initial_error >= 0.5 * error
    )
    if reference_is_broken:
        issues.append(
            f"your validation REFERENCE is the broken part, not the scheme: it "
            f"disagrees with the scheme at t=0 by {initial_error:g} while the "
            f"tolerance is {tolerance:g}, and the scheme necessarily starts from "
            f"the initial condition it was given. Correct the reference (check its "
            f"eigenvalues and series coefficients) before changing any scheme code"
        )

    # A diffusion conductance that is too large equilibrates the profile almost
    # immediately, so the scheme comes out nearly FLAT across the radius where the
    # reference is strongly curved, and the error looks like a uniform offset.
    # Naming that is far more useful than "the deviation exceeds the tolerance",
    # because it points at the conductance rather than at the time step. Measured
    # in practice: a generated solver carried a stray 1/R**2 in its conduction
    # term and produced a span of 0.01 K against a reference span of 12 K with an
    # error of 7.54 K; removing that one factor took the error to 0.015 K.
    diffusion_too_fast = (
        not reference_is_broken
        and profile_span is not None
        and reference_span is not None
        and reference_span > 0.0
        and profile_span < 0.5 * reference_span
    )
    if diffusion_too_fast:
        # A flat profile has TWO possible causes, and they point at different
        # code. An over-large conductance equilibrates the interior instantly and
        # drives the whole field toward the ambient value; a boundary flux that
        # never acts leaves the field sitting at its INITIAL value forever. Only
        # the reported values can tell them apart, and naming the wrong one sends
        # the repair to the wrong code.
        stuck_at_initial = False
        if (
            profile_max is not None
            and profile_min is not None
            and initial_value is not None
            and boundary_value is not None
        ):
            drive = abs(boundary_value - initial_value)
            if drive > 0.0:
                moved = abs(profile_max - initial_value)
                moved = max(moved, abs(profile_min - initial_value))
                stuck_at_initial = moved <= max(1e-9, 1e-6 * drive)
        if stuck_at_initial:
            issues.append(
                f"your scheme's field never left its initial value: it is flat at "
                f"{profile_max:g} across the whole domain, which is the initial "
                f"value {initial_value:g} it started from and not the boundary "
                f"value {boundary_value:g} it should be driven toward. The "
                f"interior conductance is therefore not the fault — the surface "
                f"condition is not acting at all. Check the boundary row of your "
                f"system: a boundary conductance that came out zero (a coefficient "
                f"evaluated at the wrong place, a face area that is zero at the "
                f"surface, a boundary term added to the wrong index), and confirm "
                f"the surface value actually changes after one step. This is not a "
                f"resolution problem and it is not a conductance problem"
            )
        else:
            issues.append(
                f"your scheme's profile is nearly flat where the reference is curved "
                f"(span {profile_span:g} against a reference span of {reference_span:g}), "
                f"which means the diffusion conductance is too large: look for a stray "
                f"geometric factor in the conduction term (an extra 1/R**2) or a "
                f"control-volume width missing from the storage term, and check that "
                f"the face area and the node spacing are in the same units. Shrinking "
                f"the time step will not fix this"
            )

    # A scheme that reproduces the initial profile but cannot hold the exact
    # steady state has a STRUCTURAL fault, and the two tests together say which
    # kind. Measured on this project: a generated solver's moisture block built a
    # fully implicit diagonal and then ALSO added the explicit interior fluxes to
    # the right-hand side, so the fluxes were counted twice; its moisture field
    # never moved from its initial value for the whole run, and every subproblem
    # failed. Refining the grid cannot repair a flux-bookkeeping error, so naming
    # it is the difference between a repair that converges and one that does not.
    structural_fault = (
        not reference_is_broken
        and not diffusion_too_fast
        and steady_error is not None
        and tolerance is not None
        and steady_error > tolerance
        and (initial_error is None or initial_error <= tolerance)
    )
    if structural_fault:
        t0 = "within tolerance" if initial_error is None else f"{initial_error:g}"
        issues.append(
            f"your scheme does not hold the exact steady state of the validation "
            f"case: it settles {steady_error:g} away from it (tolerance "
            f"{tolerance:g}) while reproducing the initial profile to {t0}. That "
            f"combination is a STRUCTURAL fault, not a resolution problem: check "
            f"for a flux counted twice (implicit on the diagonal AND explicit on "
            f"the right-hand side), a control-volume factor in the wrong place, or "
            f"a boundary flux applied to the wrong area. Refining the grid or the "
            f"time step cannot fix it"
        )

    if issues:
        detail = (
            "The solver's own scheme validation did not pass: "
            + "; ".join(issues)
            + (f" (case: {case})" if case else "")
        )
        if not reference_is_broken and not diffusion_too_fast and not structural_fault:
            detail += (
                ". Before rewriting the scheme, validate the REFERENCE: measure the "
                "deviation between scheme and reference at t=0 and report it as "
                "`initial_profile_error`. A reference that does not reproduce the "
                "initial profile is the more likely fault."
            )
        return VerificationCheck(
            name="scheme_self_check_passed",
            category="correctness",
            status=CheckStatus.FAIL,
            detail=detail,
            evidence={"self_check": reported},
        )
    return VerificationCheck(
        name="scheme_self_check_passed",
        category="correctness",
        status=CheckStatus.PASS,
        detail=(
            f"The scheme was validated against {case} with a maximum absolute "
            f"deviation of {error:g}, within the stated tolerance {tolerance:g}"
        ),
        evidence={"self_check": reported},
    )


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
        template_sheets: Optional[dict[str, list[str]]] = None,
        problem_text: str = "",
        requires_scheme_validation: bool = False,
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
            # The supplied template's header row is part of the answer, so the
            # wording has to match. Comparing only column counts let a result
            # file pass while every heading differed from the template.
            mismatched = header_mismatches(expected, headers)
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

        # 3a. A template that names its worksheets is stating where each answer
        # belongs. A file that puts the temperature field in a sheet called
        # "Sheet1" has not answered the question the template asked, and no
        # other check looks at worksheet names.
        template_sheets = template_sheets or {}
        sheet_issues: list[str] = []
        sheets_compared = 0
        for name in parsed:
            expected_sheets = template_sheets.get(name)
            if not expected_sheets:
                continue
            required = [
                s.strip() for s in expected_sheets
                if s.strip() and not GENERIC_SHEET_RE.match(s.strip())
            ]
            if not required:
                continue
            sheets_compared += 1
            try:
                produced = read_output_sheet_names(output_dir / name)
            except Exception as exc:
                sheet_issues.append(f"{name}: worksheets could not be read ({exc})")
                continue
            absent = [s for s in required if s not in produced]
            if absent:
                sheet_issues.append(
                    f"{name}: worksheet name mismatch — the template requires "
                    f"{required}, but the file has {produced} (missing {absent})"
                )
        if sheets_compared == 0:
            checks.append(VerificationCheck(
                name="output_worksheets_match_template",
                category="outputs",
                status=CheckStatus.NOT_APPLICABLE,
                detail="No supplied template named a worksheet the answer must reproduce",
            ))
        else:
            checks.append(VerificationCheck(
                name="output_worksheets_match_template",
                category="outputs",
                status=CheckStatus.FAIL if sheet_issues else CheckStatus.PASS,
                detail=(
                    "; ".join(sheet_issues) if sheet_issues
                    else f"{sheets_compared} output file(s) use the worksheets the template names"
                ),
                evidence={"expected_sheets": {
                    k: v for k, v in template_sheets.items() if k in parsed
                }},
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

        # 3c. When the problem states how often a table must be sampled, the
        # answer has to be sampled that often. A file that reports every 6 h
        # where the problem asks for every 60 s has not produced the requested
        # result, however correct the values in it are.
        time_steps = problem_time_steps(problem_text)
        grid_issues: list[str] = []
        grids_compared = 0
        for name, step in sorted(time_steps.items()):
            entry = parsed.get(name)
            if entry is None:
                continue
            rows = entry[1]
            times = [
                v for v in (_as_number(r[0]) for r in rows if r) if v is not None
            ]
            if len(times) < 2:
                grid_issues.append(
                    f"{name}: only {len(times)} time row(s), so the required "
                    f"{step:g} s interval cannot be met"
                )
                continue
            grids_compared += 1
            observed = [b - a for a, b in zip(times, times[1:])]
            wrong = [d for d in observed if abs(d - step) > 1e-6]
            if wrong:
                distinct = sorted({round(d, 6) for d in observed})
                grid_issues.append(
                    f"{name}: expected a time step of {step:g} s (the problem asks "
                    f"for a row every {step:g} s), but the file steps by "
                    f"{distinct[:4]} s over {len(times)} rows "
                    f"({times[0]:g}..{times[-1]:g} s)"
                )
        if grids_compared == 0 and not grid_issues:
            checks.append(VerificationCheck(
                name="output_time_grid_matches_problem",
                category="outputs",
                status=CheckStatus.NOT_APPLICABLE,
                detail="The problem states no sampling interval for the output files",
            ))
        else:
            checks.append(VerificationCheck(
                name="output_time_grid_matches_problem",
                category="outputs",
                status=CheckStatus.FAIL if grid_issues else CheckStatus.PASS,
                detail=(
                    "; ".join(grid_issues) if grid_issues
                    else f"{grids_compared} output file(s) are sampled at the stated interval"
                ),
                evidence={"required_steps_seconds": time_steps},
            ))

        # 3d. When the problem states a threshold the final state must reach,
        # the last row has to satisfy it. The drying time the answer reports is
        # only meaningful if the moisture actually fell below the target.
        targets = problem_final_state_targets(problem_text, required_outputs)
        target_issues: list[str] = []
        targets_compared = 0
        first_met: list[tuple[str, float]] = []
        for name, comparator, value in targets:
            entry = parsed.get(name)
            if entry is None:
                continue
            rows = entry[1]
            if not rows:
                target_issues.append(f"{name}: no data rows to check against {comparator} {value:g}")
                continue
            final_row = rows[-1]
            numbers = [
                v for v in (_as_number(c) for c in final_row[1:]) if v is not None
            ]
            if not numbers:
                target_issues.append(
                    f"{name}: the final row holds no numbers to check against "
                    f"{comparator} {value:g}"
                )
                continue
            targets_compared += 1
            worst = max(numbers)
            satisfied = worst < value if comparator == "<" else worst > value
            if satisfied:
                # When was the target FIRST met? If it is met almost at once,
                # "the final row satisfies the threshold" is vacuous: the answer
                # has not modelled the process at all, it has simply dried
                # everything immediately. Measured on this project: a solver
                # reached < 0.15 kg/kg from 2.55 within about an hour while the
                # statement says the process lasts 2-3 days, and this check
                # passed it.
                for row in rows:
                    row_numbers = [
                        v for v in (_as_number(c) for c in row[1:]) if v is not None
                    ]
                    if not row_numbers:
                        continue
                    row_worst = max(row_numbers)
                    met = row_worst < value if comparator == "<" else row_worst > value
                    if met:
                        first_met.append((name, _as_number(row[0])))
                        break
            if not satisfied:
                target_issues.append(
                    f"{name}: the problem requires the final moisture "
                    f"concentration to be strictly {comparator} {value:g} kg/kg, but "
                    f"the largest value in the final row (t={final_row[0]}) is "
                    f"{worst:.6f}"
                    ". The remedy is usually to INTEGRATE LONGER, not to change "
                    "the physics: where the problem asks for the time the drying "
                    "requires, the drying duration IS the answer, and the official "
                    "template places no limit on it. Verified on this project: an "
                    "independent solve using only the problem's own coefficients "
                    "(D = 7e-9*exp(-0.89/C), h_m = 8e-7 m/s, rho = 820, R = 0.02 m) "
                    "and attachment 1's own ambient moisture needs about 68-69 days "
                    "for EVERY point to fall below 0.15 kg/kg -- so a run that stops "
                    "at 72 h and reports a final row that misses the target has "
                    "answered a different question from the one asked. Extend the "
                    "integration until the target is met at every point, and output "
                    "that whole span on the required grid."
                )
        # 3e. The time the process actually takes has to be consistent with the
        # time the statement says it takes. Without this, a solver that dries
        # everything within minutes satisfies 3d vacuously -- which is exactly
        # what one measured run did.
        duration = problem_process_duration(problem_text)
        duration_issues: list[str] = []
        if duration is not None and first_met:
            low, high = duration
            # Generous: only flag something clearly impossible, not merely fast.
            floor_seconds = 0.2 * low
            for name, seconds in first_met:
                if seconds is not None and seconds < floor_seconds:
                    duration_issues.append(
                        f"{name}: the stated threshold is already reached at "
                        f"t={seconds:g} s, but the problem says the process lasts "
                        f"{low:g}-{high:g} s. Completing it in "
                        f"{100.0 * seconds / low:.1f}% of the stated time is not a "
                        f"fast answer, it means the process is not being modelled: "
                        f"check the transfer coefficients, the areas they act on, "
                        f"and any unit conversion between the two"
                    )
        if duration is not None and first_met:
            checks.append(VerificationCheck(
                name="output_process_duration_is_plausible",
                category="correctness",
                status=CheckStatus.FAIL if duration_issues else CheckStatus.PASS,
                detail=(
                    "; ".join(duration_issues) if duration_issues
                    else "the stated threshold is reached within the process time "
                         "the problem gives"
                ),
            ))
        # 3f. The physics itself bounds how fast moisture can leave: a surface
        # cannot exchange faster than `h_m * rho * C * A`. Unlike the solver's own
        # conservation audit, this depends only on the problem's stated constants,
        # so it cannot be talked out of. Measured on this project: a solver
        # removed 0.62 kg in 180 s, about 65x this ceiling, and every check the
        # solver wrote for itself passed.
        limits = problem_transport_limits(problem_text)
        ceiling_issues: list[str] = []
        if limits is not None and first_met and limits["initial_moisture_mass"] > 0.0:
            ceiling = limits["ceiling_rate"]
            initial_mass = limits["initial_moisture_mass"]
            initial_c = limits["initial_moisture"]
            for name, _seconds in first_met:
                entry = parsed.get(name)
                if entry is None:
                    continue
                header, table = entry[0], entry[1]
                # Not every header cell is a radius: result4's last column is the
                # label 药材表面. Requiring all of them to be numeric skipped that
                # whole file, leaving it uncovered by this check. Use the numeric
                # columns and ignore the labels.
                columns = [
                    (i, _as_number(cell))
                    for i, cell in enumerate(header[1:])
                    if _as_number(cell) is not None
                ]
                if len(columns) < 2:
                    continue
                # Cylinder annulus volume scales as r*dr, so for evenly spaced
                # radial samples the volume weight is r -- the centre sample
                # carries essentially no volume and the plain column mean is
                # NOT the volume mean.
                weights = [
                    0.0 if position == 0 else abs(float(radius))
                    for position, radius in enumerate(r for _, r in columns)
                ]
                if sum(weights) <= 0.0:
                    continue
                means: list[tuple[float, float]] = []
                for row in table:
                    t_value = _as_number(row[0])
                    if t_value is None:
                        continue
                    values = [_as_number(row[1 + i]) for i, _ in columns]
                    if len(values) != len(weights):
                        continue
                    total = 0.0
                    weight_sum = 0.0
                    for weight, value in zip(weights, values):
                        if value is None:
                            continue
                        total += weight * value
                        weight_sum += weight
                    if weight_sum > 0.0:
                        means.append((float(t_value), total / weight_sum))
                # Anchor the series at the INITIAL CONDITION, not at the first
                # output row. Measured on this project: a solver's output grid
                # began at t=660 s with the material already dry, so the fast
                # phase sat entirely before the first sample and every sampled
                # interval looked calm. Without this anchor the check reports a
                # pass for exactly the failure it exists to catch.
                if means and means[0][0] > 0.0:
                    means.insert(0, (0.0, initial_c))
                worst_rate = 0.0
                worst_pair = None
                for (t_prev, m_prev), (t_now, m_now) in zip(means, means[1:]):
                    if t_now <= t_prev:
                        continue
                    lost = (m_prev - m_now) * initial_mass / initial_c
                    rate = lost / (t_now - t_prev)
                    if rate > worst_rate:
                        worst_rate = rate
                        worst_pair = (t_prev, t_now)
                # A factor of 2 leaves room for a non-uniform profile and for the
                # equilibrium concentration, while still catching the order-of-
                # magnitude errors this exists for.
                if worst_pair is not None and worst_rate > 2.0 * ceiling:
                    ceiling_issues.append(
                        f"{name}: moisture leaves at up to {worst_rate:.4g} kg/s "
                        f"between t={worst_pair[0]:g} and t={worst_pair[1]:g}, but "
                        f"the problem's own coefficients permit at most "
                        f"{ceiling:.4g} kg/s at the surface "
                        f"(h_m * rho * C_init * A). No discretisation can remove "
                        f"moisture faster than the stated surface exchange allows, "
                        f"so a factor of {{:.0f}} is being applied somewhere it "
                        f"should not be -- check the density and the control-volume "
                        f"width in EACH term of the moisture balance, and check "
                        f"which area the transfer coefficient acts on.".format(
                            worst_rate / ceiling
                        )
                    )
        if limits is not None and first_met:
            checks.append(VerificationCheck(
                name="output_respects_stated_transport_ceiling",
                category="correctness",
                status=CheckStatus.FAIL if ceiling_issues else CheckStatus.PASS,
                detail=(
                    "; ".join(ceiling_issues) if ceiling_issues
                    else "the moisture removal rate stays within the ceiling the "
                         "problem's own coefficients allow"
                ),
            ))
        if targets_compared == 0 and not target_issues:
            checks.append(VerificationCheck(
                name="output_meets_stated_final_target",
                category="correctness",
                status=CheckStatus.NOT_APPLICABLE,
                detail="The problem states no numeric threshold for a final output row",
            ))
        else:
            checks.append(VerificationCheck(
                name="output_meets_stated_final_target",
                category="correctness",
                status=CheckStatus.FAIL if target_issues else CheckStatus.PASS,
                detail=(
                    "; ".join(target_issues) if target_issues
                    else f"{targets_compared} final row(s) satisfy the stated threshold"
                ),
                evidence={"targets": [
                    {"file": n, "comparator": c, "value": v} for n, c, v in targets
                ]},
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

        # 7. A model that integrates a differential equation has to show that its
        #    numerical scheme reproduces a case with a known answer. Without this
        #    a scheme can drop a control-volume factor or rebuild its
        #    time-derivative right-hand side from the current iterate, still exit
        #    0 with plausible numbers, and only surface as a disagreement with the
        #    independent recomputation — which says nothing about the cause.
        checks.append(_scheme_self_check(
            (outcome.summary or {}).get("self_check"),
            required=requires_scheme_validation,
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
        solver_formulas: Optional[dict[str, Any]] = None,
        failure_feedback: Optional[list[str]] = None,
        data_schema: str = "",
        solution_structure: Optional[dict[str, Any]] = None,
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
            *(
                [
                    "## FORMULAS THE SOLVER SAYS IT IMPLEMENTED (check these "
                    "character by character)",
                    "These are the empirical laws the solver program claims to have "
                    "coded, taken from its own report. A transcription error here is "
                    "the most dangerous defect in the whole solution, because the "
                    "program still runs cleanly and produces plausible numbers — no "
                    "numeric comparison can reveal it, and the solver may have made "
                    "the same slip you would.",
                    json.dumps(solver_formulas, ensure_ascii=False, indent=2),
                    "",
                    "For EACH formula above, do all of the following and report the "
                    "outcome as its own constraint verdict:",
                    "1. Locate the corresponding law in the problem statement.",
                    "2. Compare the STRUCTURE of every exponent, including whether "
                    "it is a PRODUCT or a RATIO. When a fraction inside an exponent "
                    "is flattened to plain text, `exp(-a/C)` arrives looking like "
                    "`e^-a C`; reading it as `exp(-a*C)` is a different function "
                    "that changes the answer by a factor of several. Settle it by "
                    "the reading that is unambiguous — an Arrhenius factor like "
                    "`e^{-3850/T}` is certainly a ratio, and two exponents typeset "
                    "the same way mean the same thing — and by cross-checking "
                    "against any duration, range or endpoint the statement gives "
                    "for the same quantity.",
                    "3. Check the UNITS the statement declares for each symbol, and "
                    "whether the solver's expression converts into them. A factor "
                    "like `exp(-3850/T)` requires T in KELVIN: passing Celsius makes "
                    "it about `1e-60` and silently freezes whatever it drives.",
                    "4. Evaluate the formula yourself at a state the run actually "
                    "reaches and report the number you get, so a wrong formula shows "
                    "up as a concrete disagreement rather than an opinion.",
                    "5. Distinguish an AMBIGUITY from an ERROR, and only fail an "
                    "error. If the statement's own typesetting genuinely admits two "
                    "readings and the solver states which one it used, that is a "
                    "recorded modelling assumption, not a defect: report it as such "
                    "and say which reading you would have chosen. FAIL the formula "
                    "only when the solver's expression contradicts something the "
                    "statement settles independently — a worked value, a stated "
                    "duration or range, a declared unit — or when it disagrees with "
                    "the reading you can prove from a cross-check. Measured on this "
                    "project: a solver correctly read a ratio inside an exponent "
                    "(the reading that reproduces the statement's own 2-3 day "
                    "process duration) and the verifier failed all three of its "
                    "formulas for being 'ambiguous', which blocked a run over a "
                    "problem-statement ambiguity rather than a solver error.",
                    "If the solver reported that its code differs from its intent, "
                    "treat the code as the truth and the intent as the claim to test.",
                    "6. Report a verdict PER FORMULA in your JSON, as "
                    '`evidence["formula_verdicts"]`, mapping each symbol to a short '
                    'string: "pass" when the solver transcribed it correctly, or a '
                    '"fail: <what differs>" when it did not. Your check `status` '
                    "must agree with those verdicts: if every entry passes, the "
                    "check passes too. Measured on this project: a verifier set a "
                    "check to FAIL while its detail said the formulas were "
                    "transcribed correctly, which blocked a run on a verdict that "
                    "contradicted its own evidence.",
                    "",
                ]
                if solver_formulas
                else []
            ),
            "## FILES AVAILABLE",
            f"- /workspace/input/ contains: {available_inputs}",
            f"- /workspace/output/ contains the solution files: {required_outputs}",
            "",
            *(
                [
                    "## SOLUTION FILE STRUCTURE (authoritative — do not guess it)",
                    "The solution files have exactly this shape. Trust it over "
                    "anything you would otherwise infer from a name or a range.",
                    json.dumps(solution_structure, ensure_ascii=False, indent=1),
                    "- The FIRST column of every such sheet is the TIME axis, not a "
                    "field value. Exclude it before you compute a value range: "
                    "including it makes every sheet span 0..t_end, so no sheet "
                    "matches any range you test and every quantity you derive comes "
                    "out as None.",
                    "- Do NOT classify or filter a sheet by its NAME. Decide what a "
                    "sheet holds from the FIELD columns (column 2 onwards), the "
                    "header row above, and the problem statement.",
                    "- `first_data_row` and `last_data_row` are that sheet's real "
                    "values (the time column first). Use them to confirm what you are "
                    "reading: a sheet whose first data row starts near the initial "
                    "MOISTURE (for example 2.55 kg/kg) holds moisture, while one "
                    "starting near the initial TEMPERATURE (for example 28 C) holds "
                    "temperature. Reading the temperature sheet while looking for "
                    "moisture is invisible from the name and yields a nonsense "
                    "verdict such as 'the initial moisture is 28.0'.",
                    "- A sheet you identified by name alone is a guess. State in "
                    "`evidence` which values convinced you, so a wrong choice is "
                    "visible instead of being reported as a solver error.",
                    "",
                ]
                if solution_structure
                else []
            ),
            "## FILE HANDLING (mandatory)",
            "- Never hardcode a file name you were not given. Discover the actual "
            "files at runtime, e.g. `sorted(os.listdir('/workspace/input'))`, and "
            "pick the one you need from that listing.",
            "- If a file you expect is missing, report the check as NOT_APPLICABLE "
            "with the listing you observed in `detail`. Do not crash, and do not "
            "invent a path.",
            "",
            "## LOCATING A QUANTITY INSIDE THE SOLUTION FILES (mandatory)",
            "- The problem statement says what each result file holds. Use IT to "
            "decide which quantity a file contains, then read that file's worksheets "
            "directly.",
            "- Do NOT select worksheets by whether their NAME contains 水, 浓度, "
            "moisture or conc. Templates are inconsistent about this: result1 and "
            "result2 may name their sheets 温度 / 水分浓度 while result3 and result4 "
            "have only Sheet1. A name filter finds nothing on those files, so every "
            "quantity you derive from them comes out as None — which is "
            "indistinguishable from the solver having written nothing at all.",
            "- Read EVERY sheet of every result file and decide what a sheet holds "
            "from its HEADER ROW, its value range and its row count. A column of "
            "concentrations in the same units as the problem's initial moisture is "
            "the moisture field, whatever its sheet happens to be called.",
            "- Reporting that a quantity is MISSING is a strong claim, and it is "
            "usually wrong. Measured on this project: a verifier reported `could "
            "not identify moisture sheet` for result3.xlsx and result4.xlsx and the "
            "run was blocked on it, while both files in fact held a worksheet named "
            "`水分浓度` with 17991 and 17999 rows and the full moisture table, whose "
            "final row read 0.0499 kg/kg against a required 0.15. The quantity was "
            "present and correct; only the verifier's expectation about the sheet's "
            "name was wrong. So before you ever report a missing quantity, you must "
            "have listed EVERY worksheet in the file and said, for each one, why it "
            "is not the quantity you want -- by its values, not its name. Put that "
            "enumeration in `evidence`. A missing-quantity verdict with no such "
            "enumeration is not evidence.",
            "- Never let a lookup fail silently into None. If you cannot identify a "
            "quantity, report that check as NOT_APPLICABLE with what you did observe, "
            "instead of emitting a verdict built on None.",
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
            "2c. Where a threshold applies matters as much as the threshold. A "
            "value the problem states for the END of a process (for example a "
            "final moisture concentration below 0.15 kg/kg, or a drying time by "
            "which that holds) applies to the FINAL row of the table, or to the "
            "rows from the drying time onwards — never to the whole table. Taking "
            "a maximum over every row including t=0 tests the INITIAL condition "
            "against a final target, which no correct answer can ever satisfy, and "
            "it is how a correct solution gets reported as failing. Conversely a "
            "constraint that holds for all t must not be checked on the last row "
            "alone.",
            "2d. Check physical bounds in the problem's OWN units. A temperature the "
            "problem gives in degrees Celsius (an initial 28 C, an ambient 60 C) is "
            "not subject to an absolute-zero bound of 273.15: that number is a "
            "Kelvin threshold, and applying it to Celsius data reports a correct "
            "answer as failing. Read the units the problem uses and hold the "
            "solution to those, and say in `evidence` which unit you assumed.",
            "2e. A constraint about a DERIVATIVE or a limit at a boundary point "
            "cannot be evaluated by differencing the output table's sample "
            "columns, and doing so reports a correct answer as failing. Measured "
            "on this project: a verifier checked the symmetry condition that the "
            "radial temperature gradient vanishes at the centre by differencing "
            "the first two columns of the output, which are sampled at r = 0 and "
            "r = 0.1 cm. It obtained `max|dT/dr| at center = 8.8000e+00` and "
            "reported a violation. The gradient at a sample 0.1 cm from the "
            "centre is genuinely non-zero -- that is what the smooth solution "
            "looks like -- so the check could not have passed for ANY correct "
            "answer. If a constraint is a derivative, a gradient, or a limiting "
            "value, check it only against a quantity the solution itself reports, "
            "or mark the check NOT_APPLICABLE and say that the output sampling "
            "cannot express it. Never turn a finite difference across coarse "
            "sample points into evidence that the solution is wrong.",
            "2f. Read the row a statistic's NAME asks for. When you recompute a "
            "quantity the solver called `..._final...`, `final_...`, or `last...`, "
            "take it from the FINAL row of the table. Reducing the whole column "
            "with `min`/`max` instead reaches back to t=0 and answers a different "
            "question. Measured on this project: recomputing a statistic named "
            "`problem1_final_min_T_C` as the minimum over every row returned "
            "28.0 -- exactly the INITIAL temperature, which is the coldest the "
            "material ever is -- against the solver's 33.5765, and the run was "
            "blocked on a 16.6% disagreement that was created by the reduction, "
            "not by either answer. A minimum over ALL rows and a minimum over the "
            "FINAL row are different statistics; say in `evidence` which row or "
            "rows you used.",
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
            "- Report PHYSICAL quantities in `statistics`: the things the problem "
            "asks for, such as a drying time, a final concentration or a "
            "temperature. Do NOT report implementation details — grid point counts, "
            "cell counts, step counts, iteration counts, or the number of columns in "
            "a file. Two correct programs may choose different discretisations, so "
            "comparing those produces a disagreement that is not one. Observed in "
            "practice: a solver using 41 internal grid points and writing 21 radius "
            "columns was reported as disagreeing with a verifier that counted the 21 "
            "columns, which blocked a run whose answer was otherwise sound.",
            "- Validate your OWN recomputation before you report it, and never "
            "report a number you have not sanity-checked. You are a generated "
            "program too, and an unstable scheme diverges: measured on this "
            "project, a verifier's own recomputation returned 4.877e+232 for a "
            "temperature that lies between 28 and 60 degrees, and every key it "
            "reported was then treated as the solver disagreeing by 100%. That "
            "sent the repair loop after a solver whose numbers were fine. Before "
            "printing a recomputed value, check it against the physical range the "
            "problem implies (a moisture concentration between 0 and its initial "
            "value, a temperature between the initial and the air temperature). "
            "Prefer an implicit scheme over an explicit one, keep the time step "
            "inside the stability limit, and if a quantity cannot be recomputed "
            "stably, OMIT that key rather than reporting a diverged value.",
            "- READ THE INPUT FILES THAT ARE ACTUALLY THERE. List the input "
            "directory and parse what you find, rather than assuming a format or "
            "a filename: on this project the attachments arrive as CSV extracts "
            "(附件1.csv, 附件2.csv), not as the original .xlsx workbooks, so "
            "pd.read_csv is right here and pd.read_excel would fail. Before "
            "trusting any number you compute, check that you actually loaded the "
            "records you expect: if your row counts are zero, STOP and report "
            "that, rather than publishing checks computed on empty input. "
            "- A TIME AT WHICH A CONDITION IS FIRST MET IS NOT THE LAST "
            "TIMESTAMP IN THE FILE. Measured twice on this project: a verifier "
            "reported the drying time as 72.0 h when the file's grid ended at "
            "259200 s (= 72 h), and again as 71.4166667 h when the grid ended at "
            "257100 s (= 71.4166667 h). Both were the file's final time, not a "
            "crossing. When you report `t_dry`, a settling time, or any "
            "'the time at which X first happens', scan the rows and report the "
            "FIRST time at which the condition actually holds. If it never holds "
            "within the data, then your finding is that it never holds: say so, "
            "or OMIT the key. Reporting the last timestamp produces a value that "
            "looks precise and is guaranteed wrong, and it will be compared "
            "against the solver's number as though it meant the same thing.",
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


# Two independently written programs agree on a number when they agree to the
# precision the verifier reported, or to within 0.1% of it, or to 1e-6 absolute
# near zero — whichever is loosest.
#
# An absolute-only tolerance of 1e-6 accused a verifier that printed its results
# to 4 decimal places of disagreeing with a solver that printed full precision:
# 33.66469441752457 against 33.6647 differs by 5.6e-6, which is rounding, not a
# disagreement. Every "mismatch" in that report was this artifact, and the false
# accusation blocked a run whose two programs actually agreed throughout.
STATISTIC_REL_TOLERANCE = 1e-3

# OPEN QUESTION, deliberately NOT resolved here (round 25). This threshold
# rejected a 0.9% difference between two independently discretised PDE solves:
# runs/cumcm-2026-a2026-auditfix reported a drying time of 43.5 h where the
# verifier recomputed 43.1167 h -- strong corroboration rather than a
# disagreement -- and that FAIL blocked the run. Raising the blanket tolerance
# to 5% fixes that case but BREAKS a requirement established in an earlier
# round, verified by two existing tests: when a verifier reports 37.05 to two
# decimals the true value lies in [37.045, 37.055], so a solver value of
# 36.981875 is genuinely different and MUST fail. Rounding-precision comparison
# and discretisation-noise tolerance are two DIFFERENT mechanisms that this one
# constant currently conflates. Resolving it properly means tolerating noise
# only for quantities that are themselves the output of a numerical
# integration while keeping precision comparison strict for everything else --
# a discriminator the host does not yet have. Recorded rather than guessed at.

STATISTIC_ABS_TOLERANCE = 1e-6
# Two independently written programs can disagree for a real reason, but a
# disagreement of many orders of magnitude is not a disagreement about the
# answer: one of the two computations diverged, and comparing the two decides
# nothing. Measured on the real A problem, a generated verifier's own
# recomputation produced 4.87729701306026e+232 for a temperature that lies
# between 28 and 60 C, and the comparison then reported the SOLVER as wrong
# ("differ by 100.0%") on every key, which sent the repair loop after a solver
# whose numbers were entirely plausible.
IMPLAUSIBLE_RATIO = 1e12
# Generous: a scheme with adaptive steps and a moving boundary will not balance
# to machine precision, and this exists to catch factor errors (order 1), not to
# grade discretisation quality.
CONSERVATION_RELATIVE_TOLERANCE = 5e-2

# A weaker threshold for the case where the two sides differ by far more than
# any genuine modelling disagreement can explain. Two correct programs cannot
# disagree about the same physical quantity by five orders of magnitude, so one
# of them is broken; the larger side is the one that has run away. Measured on
# the real A problem: a verifier recomputed a final moisture concentration of
# 33960 kg/kg where the solver reported 0.1469 and the material starts at 2.55,
# a ratio of 2.3e5 -- absurd as a concentration, yet below IMPLAUSIBLE_RATIO, so
# it was reported as the solver disagreeing and blocked the run with three
# checks. Attributing it to the verifier cannot let a wrong answer through: the
# affected checks are downgraded to NOT_RUN, which still fails the report, and
# the original finding is preserved in the detail.
DIVERGED_RATIO = 1e5

# A validation deviation this large is a blow-up, not an accuracy shortfall, and
# saying "exceeds the tolerance" about it tells the solver nothing.
DIVERGED_ABSOLUTE = 1e6
# A check can also rest on a diverged number in its prose rather than in the
# statistics. Measured on the real A problem, a generated verifier reported
# "max |dT| at final t=1800s = 2.6713...e+155 C" as its counterexample, and that
# FAIL blocked the run even though the comparison layer had already established
# that the verifier's own recomputation was the diverged side. No physical
# quantity in a competition problem is this large, so a check quoting one is not
# evidence about the solver and must not be reported as one.
IMPLAUSIBLE_ABS_LIMIT = 1e30


def _implausible_numbers(text: str) -> list[float]:
    """Numbers in ``text`` too large to be a physical quantity."""
    found: list[float] = []
    for match in re.finditer(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text or ""):
        try:
            value = float(match.group())
        except ValueError:
            continue
        if abs(value) >= IMPLAUSIBLE_ABS_LIMIT:
            found.append(value)
    return found


FORMULA_VERDICTS_KEY = "formula_verdicts"


def unaccountable_formula_verdict(check: "VerificationCheck") -> Optional[str]:
    """A FAIL whose own per-formula verdicts all pass is not evidence.

    The independent verifier is generated code too, and it can contradict
    itself exactly as the solver can. Measured on this project: a verifier
    emitted a check with status FAIL whose entire detail read
    "D_q1=e^-0.89/C ratio; D_q23/D_q4 use exp(-3850/T_K) Arrhenius ratio (T in
    Kelvin)" -- a statement that the formulas were transcribed CORRECTLY -- and
    blocked the run on it. Reading the status in place of the detail is the same
    error as trusting a solver's bare `passed` flag, so when the verifier
    reports per-formula verdicts and every one of them passes, the host does not
    treat the FAIL as evidence.

    Returns the check's own detail when the verdict is unaccountable, else None.
    """
    if check.status is not CheckStatus.FAIL or check.category != "independent":
        return None
    evidence = check.evidence or {}
    verdicts = evidence.get(FORMULA_VERDICTS_KEY)
    if not isinstance(verdicts, dict) or not verdicts:
        return None
    rendered = " ".join(str(item) for item in verdicts.values()).lower()
    if "fail" in rendered or "mismatch" in rendered or "differ" in rendered:
        return None
    return check.detail


def unevidenced_by_diverged_recomputation(
    check: VerificationCheck,
) -> Optional[float]:
    """The implausible magnitude this check rests on, if it rests on one."""
    haystack = check.detail or ""
    if check.evidence:
        try:
            haystack += " " + json.dumps(check.evidence, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            haystack += " " + repr(check.evidence)
    for value in _implausible_numbers(haystack):
        return value
    return None


def _magnitude(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return abs(float(value))


def _diverged_pair(claimed: Any, recomputed: Any) -> Optional[bool]:
    """Whether a pair shows a diverged computation rather than a real gap.

    Returns True when the RECOMPUTED value is the implausible one, False when the
    claimed value is, and None when the pair looks like an ordinary disagreement.
    """
    a = _magnitude(claimed)
    b = _magnitude(recomputed)
    if a is None or b is None:
        return None
    if a != a or b != b or a == float("inf") or b == float("inf"):
        # A non-finite side is always the broken one; if both are, blame the
        # recomputation, which is the side being asked to prove itself.
        return not (b == b and b != float("inf"))
    if a == 0.0 or b == 0.0:
        return None
    if max(a, b) / min(a, b) < DIVERGED_RATIO:
        return None
    return b > a


# Statistics that describe HOW a program discretised the problem rather than
# what it computed. Two correct programs may legitimately choose different
# grids, so a difference here is not a disagreement about the answer. Observed
# in practice: a solver using 41 internal grid points and writing 21 radius
# columns was reported as disagreeing with a verifier that counted those 21
# columns, and that blocked a run whose answer was otherwise sound.
_STRUCTURAL_STATISTIC_KEYS = frozenset({
    "grid_points", "grid_size", "n_grid", "num_grid", "n_points", "num_points",
    "cells", "cell_count", "n_cells", "num_cells", "steps", "step_count",
    "n_steps", "num_steps", "iterations", "iteration_count", "n_iter",
    "num_iterations", "rows", "row_count", "n_rows", "num_rows", "cols",
    "col_count", "columns", "n_cols", "num_columns", "dof", "n_dof",
    "unknowns", "num_unknowns",
})


def _is_structural_statistic(path: str) -> bool:
    """Whether a statistic names an implementation detail, not a result."""
    return path.rsplit(".", 1)[-1].strip().lower() in _STRUCTURAL_STATISTIC_KEYS


def _reported_decimals(value: Any) -> Optional[int]:
    """How many decimal places a reported number carries.

    A verifier that prints 4 decimals has discarded everything beyond them, and
    at small magnitudes that is a large relative error: 0.0279489 printed as
    0.0279 differs by 0.18%, which says nothing about disagreement.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return 0
    text = repr(float(value))
    if "e" in text or "E" in text:
        # Scientific notation does not expose a precision to compare against.
        return None
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def _statistics_disagree(
    claimed: float, recomputed: Any, tolerance: float, rel_tolerance: float
) -> bool:
    """Whether two reported numbers differ by more than their reporting allows.

    Comparison happens at the precision the verifier actually reported, because
    that precision is a bound on how far its value can have been rounded. A flat
    relative tolerance cannot do this: it is too tight for a small value printed
    to 4 decimals and too loose for a large one, so it both invents
    disagreements and hides real ones.
    """
    claimed_value = float(claimed)
    recomputed_value = float(recomputed)
    difference = abs(claimed_value - recomputed_value)
    scale = max(abs(claimed_value), abs(recomputed_value))
    allowed = max(tolerance, rel_tolerance * scale)
    places = _reported_decimals(recomputed)
    if places is not None:
        # Rounding to `places` decimals moves a value by at most half a unit in
        # the last place.
        allowed = max(allowed, 0.5 * 10.0 ** (-places) + 1e-12)
    return difference > allowed


def _compare_statistic_values(
    claimed: Any,
    recomputed: Any,
    path: str,
    tolerance: float,
    mismatches: list[str],
    counter: list[int],
    rel_tolerance: float = STATISTIC_REL_TOLERANCE,
    skipped: Optional[list[str]] = None,
    diverged: Optional[dict[str, list[str]]] = None,
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
                    tolerance, mismatches, counter, rel_tolerance, skipped, diverged,
                )
        return

    # bool is a subclass of int; comparing True with 1 would be wrong here.
    if isinstance(claimed, bool) or isinstance(recomputed, bool):
        counter[0] += 1
        if claimed != recomputed:
            mismatches.append(f"{path}: solver={claimed} independent={recomputed}")
        return

    if isinstance(claimed, (int, float)) and isinstance(recomputed, (int, float)):
        if _is_structural_statistic(path):
            # A different grid is a different choice, not a different answer.
            if skipped is not None:
                skipped.append(path)
            return
        diverged_side = _diverged_pair(claimed, recomputed)
        if diverged_side is not None:
            # One computation blew up. Comparing it with the other decides
            # nothing, and calling the surviving side "wrong" sends the repair
            # loop after the wrong component.
            if diverged is not None:
                diverged["verifier" if diverged_side else "solver"].append(path)
            return
        counter[0] += 1
        if _statistics_disagree(claimed, recomputed, tolerance, rel_tolerance):
            scale = max(abs(float(claimed)), abs(float(recomputed)), 1e-12)
            relative = abs(float(claimed) - float(recomputed)) / scale
            mismatches.append(
                f"{path}: solver={claimed} independent={recomputed} "
                f"(differ by {relative:.1%})"
            )
        return

    counter[0] += 1
    if claimed != recomputed:
        mismatches.append(f"{path}: solver={claimed!r} independent={recomputed!r}")


def compare_statistics(
    solver_stats: dict[str, Any],
    independent_stats: dict[str, Any],
    tolerance: float = STATISTIC_ABS_TOLERANCE,
    rel_tolerance: float = STATISTIC_REL_TOLERANCE,
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
    skipped: list[str] = []
    diverged: dict[str, list[str]] = {"verifier": [], "solver": []}
    counter = [0]
    for key, claimed in solver_stats.items():
        if key not in independent_stats:
            continue
        _compare_statistic_values(
            claimed, independent_stats[key], str(key), tolerance, mismatches,
            counter, rel_tolerance, skipped, diverged,
        )
    compared = counter[0]

    if diverged["verifier"]:
        # The verifier's own recomputation blew up. Its numbers cannot be used as
        # the reference, so this is not a verdict on the solver's answer.
        names = ", ".join(sorted(diverged["verifier"])[:6])
        return VerificationCheck(
            name="independent_statistics_agree",
            category="independent",
            status=CheckStatus.FAIL,
            detail=(
                f"the independent verifier's own recomputation diverged and cannot "
                f"be used as a reference: {names} (the claimed values are finite "
                f"and plausible, the recomputed ones are not). Regenerate the "
                f"verifier with a numerically stable scheme before treating any "
                f"disagreement as the solver's fault"
            ),
            evidence={
                "compared": compared,
                "skipped_structural": sorted(skipped),
                "verifier_diverged": sorted(diverged["verifier"]),
            },
        )

    for name in sorted(diverged["solver"]):
        mismatches.append(
            f"{name}: the solver's own value is not physically plausible while the "
            f"recomputed one is, so the solver's computation diverged"
        )

    if compared == 0 and not diverged["solver"]:
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
            evidence={"compared": compared, "skipped_structural": sorted(skipped)},
        )
    return VerificationCheck(
        name="independent_statistics_agree",
        category="independent",
        status=CheckStatus.PASS,
        detail=(
            f"{compared} statistic(s) independently reproduced"
            + (
                f"; {len(skipped)} implementation detail(s) not compared "
                f"({', '.join(sorted(skipped)[:5])})"
                if skipped
                else ""
            )
        ),
        evidence={"compared": compared, "skipped_structural": sorted(skipped)},
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
