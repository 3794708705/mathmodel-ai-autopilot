"""MathModel AI — CUMCM Autopilot: solver code generation and real execution.

The mathematical model is the source of truth. The generated program is
executed for real in the Docker sandbox; every number used later in the
paper comes from that execution, never from model prose.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from mathmodel.agents.base import AgentError, AgentStatus
from mathmodel.domain.math_model import MathematicalModel
from mathmodel.routing.profile import TaskProfile, TaskType
from mathmodel.routing.router import ModelRouter
from mathmodel.sandbox.backend import ExecutionRecord, ExecutionStatus, SandboxLimits
from mathmodel.sandbox.docker_backend import DockerSandboxBackend

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = "mathmodel-ai-autopilot:latest"
# A full solver program can exceed 20k characters; the provider default of 4096
# truncates it mid-file. 49152 is accepted by both the DeepSeek API and the
# local OpenAI-compatible gateway while still leaving headroom under their caps.
CODE_MAX_TOKENS = 49152

# Sent when a reply was cut off before the program finished. A truncated program
# is worse than a compact one: it may still parse, so it runs and silently
# produces nothing at all.
TRUNCATION_RETRY_INSTRUCTION = (
    "Your previous reply was CUT OFF before the program was complete. "
    "Rewrite the COMPLETE program from the very beginning, but make it much "
    "more compact: aim for well under 250 lines, shorter names, almost no "
    "comments, no duplicated blocks, and one reusable helper instead of "
    "repeated code. Do not restate the problem. It must still end with the "
    "required final line printing the JSON summary."
)


def _code_looks_complete(code: str) -> bool:
    """A program is usable only if it parses and still prints its summary.

    Truncated replies are the dangerous case: the surviving prefix often parses
    cleanly, so it runs, exits 0, and reports nothing.
    """
    if not code.strip():
        return False
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError:
        return False
    return "print(" in code[-2000:]


async def generate_program_source(
    router: ModelRouter,
    *,
    profile: TaskProfile,
    prompt: str,
    system_prompt: str,
    max_tokens: int = CODE_MAX_TOKENS,
    attempts: int = 3,
) -> str:
    """Ask for a complete program, retrying if the reply was truncated."""
    best = ""
    retry_note = ""
    # Ask for thinking to be off from the very first attempt. A long program has
    # to own the entire output budget: gateways that ignore reasoning_effort
    # otherwise spend all of max_tokens on reasoning and return zero characters
    # with finish_reason=length, which looks exactly like an empty answer.
    extra_body: Optional[dict] = {"thinking": {"type": "disabled"}}
    for _ in range(max(1, attempts)):
        response = await router.route_generate(
            profile=profile,
            prompt=prompt + retry_note,
            max_tokens=max_tokens,
            reasoning_effort="none",
            extra_body=extra_body,
            system_prompt=system_prompt,
        )
        code = _extract_code(response.content)
        truncated = response.finish_reason == "length" or not _code_looks_complete(code)
        if code and not truncated:
            return code
        if len(code) > len(best):
            best = code
        logger.warning(
            "Program reply looks incomplete (%d chars, finish_reason=%s); retrying compactly",
            len(response.content), response.finish_reason,
        )
        retry_note = (
            "\n\n" + TRUNCATION_RETRY_INSTRUCTION
            + f"\n(The previous reply was {len(response.content)} characters.)"
        )
    return best


class GeneratedProgram(BaseModel):
    """A self-contained solver program produced from the mathematical model."""

    approach: str = Field(default="", description="One paragraph: the algorithm actually implemented")
    code: str = Field(..., min_length=1, description="Complete runnable Python source")
    dependencies: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    statistics_keys: list[str] = Field(
        default_factory=list,
        description="Keys the program prints in its JSON summary",
    )


class ExecutionOutcome(BaseModel):
    """Real sandbox execution of a generated program."""

    run_id: str = ""
    status: str = ""
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    runtime_seconds: float = 0.0
    artifacts: list[str] = Field(default_factory=list)
    output_dir: str = ""
    execution_real: bool = False
    sandbox: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS.value


class SolverCodeGenerator:
    """Generates a runnable solver program from a MathematicalModel."""

    def __init__(self, router: ModelRouter):
        self._router = router

    async def generate(
        self,
        model: MathematicalModel,
        problem_text: str,
        data_schema: str,
        required_outputs: list[str],
        input_file_names: list[str],
        previous_code: Optional[str] = None,
        failure_feedback: Optional[list[str]] = None,
        output_headers: Optional[dict[str, list[str]]] = None,
    ) -> GeneratedProgram:
        prompt = self._build_prompt(
            model=model,
            problem_text=problem_text,
            data_schema=data_schema,
            required_outputs=required_outputs,
            input_file_names=input_file_names,
            output_headers=output_headers,
        )
        if previous_code and failure_feedback:
            prompt += self._repair_block(previous_code, failure_feedback)

        profile = TaskProfile.for_task_type(TaskType.CODE_GENERATION)
        # The program is requested as plain source text, not as a string field
        # inside a JSON object: escaping a several-hundred-line program into
        # JSON is where these models fail (empty object or echoed schema).
        # Reasoning is disabled because a long program must own the whole
        # output budget (measured: reasoning consumed 15031 of 32768 tokens).
        code = await generate_program_source(
            self._router,
            profile=profile,
            prompt=prompt,
            system_prompt=(
                "You are an expert scientific Python programmer for mathematical "
                "modeling competitions. You write complete, self-contained, "
                "deterministic programs that actually run and actually compute. "
                "You never hardcode results and never print numbers you did not compute. "
                "You reply with raw Python source only."
            ),
        )

        if not code.strip():
            raise ValueError(
                "Code generation returned no program text"
            )

        return GeneratedProgram(
            approach=_extract_header_field(code, "APPROACH")
            or f"{model.name} implemented as a single self-contained program",
            code=code,
            dependencies=_extract_imports(code),
            expected_outputs=list(required_outputs),
            statistics_keys=[],
        )

    # ── Prompt construction ───────────────────────────────────

    def _build_prompt(
        self,
        model: MathematicalModel,
        problem_text: str,
        data_schema: str,
        required_outputs: list[str],
        input_file_names: list[str],
        output_headers: Optional[dict[str, list[str]]] = None,
    ) -> str:
        model_view = {
            "name": model.name,
            "description": model.description,
            "model_family": model.model_family,
            "algorithm_plan": model.algorithm_plan,
            "variables": [
                {
                    "symbol": v.symbol,
                    "meaning": v.meaning,
                    "type": v.variable_type.value,
                    "domain": v.domain,
                    "indices": v.indices,
                }
                for v in model.variables
            ],
            "parameters": [
                {"symbol": p.symbol, "value": p.value, "unit": p.unit}
                for p in model.parameters
            ],
            "objectives": [
                {"name": o.name, "sense": o.sense.value, "expression": o.expression}
                for o in model.objectives
            ],
            "constraints": [
                {
                    "name": c.name,
                    "expression": c.expression,
                    "relation": c.relation.value,
                    "rhs": c.rhs,
                }
                for c in model.constraints
            ],
            "equations": [
                {"name": e.name, "expression": e.expression}
                for e in model.equations
            ],
        }

        return "\n".join([
            "Write ONE self-contained Python program that solves the competition "
            "problem by implementing the mathematical model below.",
            "",
            "## MATHEMATICAL MODEL (source of truth)",
            json.dumps(model_view, ensure_ascii=False, indent=1)[:14000],
            "",
            "## PROBLEM STATEMENT (excerpt)",
            problem_text[:8000],
            "",
            "## INPUT DATA (already parsed from the attachments)",
            data_schema[:6000],
            "",
            "## REQUIRED OUTPUT FILES",
            "\n".join(f"- /workspace/output/{name}" for name in required_outputs) or "- (none specified)",
            "",
            *(
                [
                    "## MANDATORY COLUMN HEADERS",
                    "The competition supplies a blank template for each result file. "
                    "The first row of every output file MUST be exactly these header "
                    "cells, in this order and with this exact wording. Do not rename, "
                    "reorder, translate or drop any column.",
                    json.dumps(output_headers, ensure_ascii=False, indent=2),
                    "A template whose first column is 用频装备编号 is a per-plan table: "
                    "write ONE ROW FOR EVERY plan in the input data, including the plans "
                    "you leave untouched (mark those as not cancelled and repeat their "
                    "original intervals). Do not list only the plans you changed.",
                    "",
                ]
                if output_headers
                else []
            ),
            "## EXECUTION CONTRACT (must be followed exactly)",
            "1. The program runs as `python /workspace/code/code.py` with cwd=/workspace.",
            "2. Read every input file from `/workspace/input/` (read-only).",
            f"   Available input files: {input_file_names}",
            "3. Write every output file to `/workspace/output/` (writable).",
            "4. Available libraries: numpy, scipy, pandas, openpyxl, matplotlib, "
            "and the Python standard library. There is NO network access.",
            "5. Do not read or write anything outside /workspace.",
            "6. The program MUST be deterministic (fix any random seed).",
            "7. As the LAST line of stdout, print a single JSON object on its own "
            "line describing the results, with this shape:",
            '   {"statistics": {...}, "per_subproblem": {...}, "notes": "..."}',
            "   Every number you report anywhere must appear in that JSON.",
            "",
            "## OUTPUT FORMAT (follow exactly)",
            "Reply with raw Python source only. No markdown fences, no commentary "
            "before or after the code. The first line must be a comment:",
            "# APPROACH: <one line describing the algorithm you implemented>",
            "",
            "## HARD RULES",
            "- Never hardcode a result value. Compute it.",
            "- Never hardcode an input file name you were not given: discover the "
            "actual files with `sorted(os.listdir('/workspace/input'))` and select "
            "from that listing, or use one of the exact names listed above.",
            "- If the model cannot produce a required output, write the output file "
            "anyway with the best real computation you have, and say so in `notes`.",
            "- Print progress diagnostics to stdout as plain text lines before the "
            "final JSON line.",
        ])

    @staticmethod
    def _repair_block(previous_code: str, failure_feedback: list[str]) -> str:
        return "\n".join([
            "",
            "## PREVIOUS ATTEMPT FAILED — REPAIR IT",
            "The previous program did not satisfy the requirements. Issues found:",
            *[f"  - {item}" for item in failure_feedback],
            "",
            "Previous program (for reference only — fix the actual cause):",
            "```python",
            previous_code[:12000],
            "```",
            "",
            "Return a corrected complete program as raw Python source only "
            "(no markdown fences, no commentary). Do not merely suppress the error: "
            "fix the underlying computation or data handling.",
        ])


class SandboxProgramRunner:
    """Runs a generated program in the Docker sandbox and keeps its artifacts."""

    def __init__(
        self,
        image: str = DEFAULT_SANDBOX_IMAGE,
        timeout_seconds: int = 600,
        max_memory_mb: int = 4096,
    ):
        self._image = image
        self._limits = SandboxLimits(
            timeout_seconds=timeout_seconds,
            max_memory_mb=max_memory_mb,
            max_processes=256,
            network_enabled=False,
        )

    @property
    def available(self) -> bool:
        return DockerSandboxBackend(image=self._image).available

    async def run(
        self,
        program: GeneratedProgram,
        input_files: dict[str, str],
        output_dir: str | Path,
    ) -> ExecutionOutcome:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        backend = DockerSandboxBackend(image=self._image)
        if not backend.available:
            return ExecutionOutcome(
                status=ExecutionStatus.BACKEND_UNAVAILABLE.value,
                stderr=f"Docker image '{self._image}' is not available",
                output_dir=str(output_dir),
            )

        record: ExecutionRecord = await backend.execute(
            code=program.code,
            limits=self._limits,
            input_files=input_files,
            capture_dir=output_dir,
        )

        captured = sorted(
            str(p.relative_to(output_dir))
            for p in output_dir.rglob("*")
            if p.is_file()
        )

        return ExecutionOutcome(
            run_id=record.run_id,
            status=record.status.value,
            exit_code=record.exit_code,
            stdout=record.stdout,
            stderr=record.stderr,
            runtime_seconds=record.runtime_seconds,
            artifacts=captured,
            output_dir=str(output_dir),
            execution_real=record.execution_real and not record.is_mock,
            sandbox={
                "backend": record.backend,
                "image": record.image,
                "code_hash": record.code_hash,
                "network_isolated": record.network_isolated,
                "filesystem_isolated": record.filesystem_isolated,
                "non_root": record.non_root,
                "no_new_privileges": record.no_new_privileges,
                "resource_limits_enforced": record.resource_limits_enforced,
                "artifact_containment": record.artifact_containment,
                "timed_out": record.timed_out,
            },
            summary=_parse_summary(record.stdout),
        )


def _parse_summary(stdout: str) -> dict[str, Any]:
    """Extract the last JSON object printed on its own line."""
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_OPENING_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n")
_FENCE_LINE_RE = re.compile(r"^[ \t]*```[A-Za-z0-9_+-]*[ \t]*$", re.MULTILINE)
_PY_START_RE = re.compile(
    r"^(?:#|import[ \t]|from[ \t]|def[ \t]|class[ \t]|@|\"\"\"|''')", re.MULTILINE
)
_HEADER_FIELD_RE = re.compile(r"^#\s*([A-Z_]+)\s*:\s*(.+)$", re.MULTILINE)
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def _extract_code(content: str) -> str:
    """Pull raw Python source out of a model reply.

    Replies arrive with a closed fence, an unterminated fence (the response was
    truncated before the closing marker), or no fence at all with prose around
    the source. All three must yield runnable source, because a stray ``` is a
    SyntaxError that looks like a solver bug.
    """
    text = (content or "").strip()
    if not text:
        return ""

    fenced = _CODE_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    else:
        opening = _OPENING_FENCE_RE.search(text)
        if opening:
            text = text[opening.end():]

    # An unterminated block leaves the closing marker inside the source.
    text = _FENCE_LINE_RE.sub("", text)
    text = text.strip("\n").strip()

    # Drop any prose that preceded the first line of actual source.
    start = _PY_START_RE.search(text)
    if start and start.start() > 0:
        text = text[start.start():]

    return _largest_compilable_prefix(text.strip())


def _largest_compilable_prefix(code: str) -> str:
    """Trim trailing prose so the result at least compiles.

    A program that does not parse produces no results at all, so the longest
    compilable prefix is strictly more useful. If nothing compiles the original
    text is returned unchanged and the failure surfaces as a real error.
    """
    if not code:
        return code
    try:
        compile(code, "<generated>", "exec")
        return code
    except SyntaxError:
        pass

    lines = code.split("\n")
    for end in range(len(lines) - 1, 0, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        if not candidate:
            continue
        try:
            compile(candidate, "<generated>", "exec")
            return candidate
        except SyntaxError:
            continue
    return code


def _extract_header_field(code: str, field: str) -> str:
    """Read a `# FIELD: value` marker from the program header."""
    for name, value in _HEADER_FIELD_RE.findall(code[:2000]):
        if name.upper() == field.upper():
            return value.strip()
    return ""


def _extract_imports(code: str) -> list[str]:
    """Third-party modules the program imports, for the support package."""
    stdlib = {
        "os", "sys", "json", "math", "re", "csv", "itertools", "collections",
        "random", "time", "datetime", "pathlib", "typing", "functools",
        "operator", "copy", "heapq", "bisect", "statistics", "warnings",
        "subprocess", "traceback", "decimal", "fractions", "textwrap",
        "__future__", "base64", "hashlib", "dataclasses", "enum", "abc",
    }
    found = {name for name in _IMPORT_RE.findall(code) if name not in stdlib}
    return sorted(found)
