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

# How much of the previous program the repair prompt shows back. The generated
# solvers run to 22k-36k characters, and the earlier 12k cut hid the tail: the
# model was asked to "return a corrected complete program" while seeing only its
# first third, so each repair fixed the reported issue and silently introduced a
# new defect in the part it could no longer read — a dropped centre node,
# off-grid time rows, a summary line that stopped being printed.
REPAIR_CODE_MAX_CHARS = 60000

# How long the generated program is allowed to run in the sandbox. This is a
# real acceptance constraint on the program, so the code generator is told about
# it: a solver that cannot finish is not a solver. Measured in practice: a
# generated implicit Crank-Nicolson program with Newton iteration per step was
# killed at exactly this limit on all three attempts, having printed nothing at
# all, because the model had no idea a budget existed.
PROGRAM_TIME_BUDGET_SECONDS = 600

# The sandbox's memory cap, for the same reason as the time budget: it is a real
# acceptance constraint the program cannot discover for itself. Measured in
# practice: two of three attempts on the real A problem were OOM-killed after
# ~160 s while the third succeeded in 14 s, because the model vectorised over
# both space and time and materialised the whole history.
PROGRAM_MEMORY_BUDGET_MB = 4096

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
                    "Some of these header rows contain a cell that is only '…' (or "
                    "'...'). That cell is a placeholder for a run of columns the "
                    "template did not print, and it has to be expanded like this: "
                    "keep every cell BEFORE the ellipsis exactly as printed, keep "
                    "every cell AFTER it exactly as printed — including the last "
                    "column, which is often a label such as 药材表面 and NOT the "
                    "number it denotes — and fill the omitted middle with numbers "
                    "continuing the arithmetic progression of the numbers printed "
                    "before the ellipsis, at that same step. The output needs at "
                    "least one column in place of the ellipsis, and the progression "
                    "must stop one step before the number that follows it. Never "
                    "replace a printed label with the number it stands for, and "
                    "never renumber the columns the template already printed.",
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
            '   {"statistics": {...}, "per_subproblem": {...}, '
            '"self_check": {...}, "formulas": {...}, "notes": "..."}',
            "   Every number you report anywhere must appear in that JSON.",
            "   `formulas` maps each symbol you implemented from an empirical law in "
            "the statement to the expression you actually wrote in code, as a string "
            "— for example `D_q1` -> `7e-9 * exp(-0.89 / C)` and `D_q3` -> "
            "`2.4e-3 * exp(-0.45 / C) * exp(-3850 / T_K)`. "
            "Report what the code DOES, not what you intended: "
            "if the two differ, say so in `notes`. An independent verifier compares "
            "these strings against the problem statement, and a mismatch between the "
            "statement and your code is the single most expensive defect in this "
            "kind of program because the run still produces plausible numbers.",
            "",
            "## TRANSCRIBING EMPIRICAL FORMULAS (hard rules)",
            "The problem statement gives property laws such as "
            "`D = 2.4e-3 * exp(-0.45 / C) * exp(-3850 / T)`. These are the physics "
            "of the whole model, so a transcription slip does not look like a bug — "
            "the program runs cleanly and produces plausible-looking numbers that are "
            "wrong by a large factor.",
            "- Expect a RATIO inside an exponent, and give the fraction its "
            "numerator and denominator. The laws in this problem put `-0.45/C` and "
            "`-3850/T` in exponents. When such a formula reaches you as flat text the "
            "fraction collapses: `exp(-0.45 / C)` arrives looking like `e^-0.45 C`, "
            "and reading that as a product (`exp(-0.45 * C)`) changes the answer by a "
            "factor of several. Measured on this project: an independent host "
            "computation of the same problem gives a drying time of 57.00 h for the "
            "quotient against 16.79 h for the product.",
            "- Settle an ambiguous exponent by finding the reading that is NOT "
            "ambiguous. This statement typesets the same form twice, and one of them "
            "is unmistakable: `e^{-3850/T}` is an Arrhenius factor, so `3850` is a "
            "numerator and `T` is a denominator. In the statement's typesetting the "
            "numerator sits high and the denominator low, and `-0.45` over `C` shows "
            "exactly that arrangement — the same shape as `3850` over `T`. Two "
            "identically-typeset exponents in one statement mean the same thing: "
            "decide the convention once, state it in a comment at the top of the "
            "file, and apply it to every formula. Do not read one as a product and "
            "another as a quotient.",
            "- Cross-check your reading against every quantitative statement the "
            "problem makes about the same quantity. This problem states how long the "
            "process takes (the drying process generally lasts 2-3 days) and its "
            "shrinkage attachment tabulates the radius out to 72 h. A reading that "
            "dries the material in 0.70 days contradicts both, while the quotient "
            "gives 2.38 days and agrees with both. Before you commit to a reading of "
            "an ambiguous formula, test it against the problem's own numbers; if two "
            "readings both survive, say which you chose and why in your summary.",
            "- Print the property values you actually use, and check the trend is "
            "the one your formula implies. For these laws the diffusivity FALLS as "
            "the material dries, which is why drying takes days and why the moisture "
            "approaches the ambient level asymptotically instead of crossing it "
            "early. If your numbers show the opposite trend to the function you "
            "wrote, you have not computed the function you wrote.",
            "- Know that an Arrhenius factor amplifies a temperature error, and "
            "keep every field inside its physical range. `exp(-3850/T)` is "
            "extremely sensitive: a modest error in `T` moves the diffusivity by "
            "orders of magnitude, and a temperature that drifts to an unphysical "
            "value silently freezes the mass transfer. Measured on this project: a "
            "generated solver's temperature field went negative in Celsius, which "
            "drove `exp(-3850/T)` toward zero, so its moisture stayed at its "
            "initial value for the whole run and every subproblem failed — the "
            "reported fault looked like mass transfer, but the cause was heat. "
            "Clip every physical field to its admissible range after each step "
            "(`T` above absolute zero and within the process bounds, `C` in "
            "`[0, C_initial]`), and if a clip actually binds, print it: a clamp that "
            "fires is telling you the scheme diverged, and silently clamping turns a "
            "diverged run into a plausible-looking wrong answer.",
            "- Respect the units the statement declares for each symbol. A factor "
            "like `exp(-3850/T)` is a strongly varying function of temperature and "
            "the statement gives `T` in KELVIN. Passing a Celsius temperature turns "
            "`exp(-3850/28)` into about `1e-60`, so the diffusivity collapses to "
            "zero and the moisture field does not move at all. Measured on this "
            "project: a generated solver wrote `exp(-3850.0 / np.maximum(T, 1.0))` "
            "and its moisture never left its initial value. The `maximum(T, 1.0)` "
            "guard is itself the tell — a temperature in Kelvin is never near 1, so "
            "guarding against small values means the variable is in Celsius. Convert "
            "explicitly (`T_K = T_C + 273.15`) at the point of use and name the "
            "variable so the unit is visible.",
            "",
            "## NUMERICAL SCHEME CORRECTNESS (hard rules)",
            "These are the failure modes that have actually produced wrong answers "
            "on this project. A program that breaks one of them runs cleanly, exits "
            "0, and reports entirely plausible numbers, so nothing downstream can "
            "catch it for you.",
            "- Discretize consistently. In a finite-volume / control-volume balance "
            "EVERY term must carry the same control-volume measure. If you divide a "
            "face flux by the cell's radial (or axial) measure, divide the storage "
            "term by that same measure. Never mix a per-node coefficient with a "
            "per-volume one, and never drop the cell width from an interior "
            "diffusion coefficient: it silently scales the whole scheme.",
            "- Check the conduction term dimensionally before you trust it. The "
            "conductance between two neighbouring nodes is `k * A_face / (node "
            "spacing)`, with the face area and the spacing expressed in the SAME "
            "length unit, and the ratio of that conductance to the storage term "
            "`rho * cp * volume / dt` must be dimensionless. A stray geometric "
            "factor — an extra `1/R**2` or `R**2` on the conductivity, a face area "
            "in one unit and a spacing in another — leaves the program running "
            "cleanly, exits 0, and merely scales the diffusion rate. Measured on "
            "this project: a generated solver carried an extra `1/R**2` in its "
            "conduction term, which made diffusion roughly 2500x too fast, flattened "
            "the radial profile to within 0.01 K of uniform where the correct "
            "profile spanned 12 K, and produced a 7.54 K error; removing that one "
            "factor took the error to 0.015 K. The mechanical tell is profile "
            "SHAPE, not magnitude: if your centre and surface values come out nearly "
            "equal at a time when the reference separates them, a geometric factor "
            "is wrong. Report that span (see `profile_span` below) and fix the "
            "factor — do not respond by shrinking the time step, which cannot "
            "change a dimensionless scaling error.",
            "- Linearize correctly. Snapshot the previous time level into its own "
            "variable BEFORE the iteration loop (e.g. `U_prev = U.copy()`) and read "
            "the time-derivative right-hand side ONLY from that snapshot. In the "
            "implicit step A(U^(k)) * U^(k+1) = b(U_prev), only the COEFFICIENTS may "
            "use the current iterate; the right-hand side must never use it. Writing "
            "the right-hand side from the same array you overwrite with each iterate "
            "makes the storage term cancel itself at the fixed point, so the loop "
            "converges to the STEADY STATE of the equation at every time step and "
            "the transient disappears entirely. Two mechanical tells: a snapshot "
            "variable that is assigned and then never read, and an answer that does "
            "not change when you halve the time step.",
            "- Do not count a flux twice. In an implicit step the right-hand side "
            "must carry the PREVIOUS time level through the STORAGE term only. The "
            "standard backward-Euler row is "
            "`(storage + condL + condR) * U_new[i] - condL * U_new[i-1] "
            "- condR * U_new[i+1] = storage * U_prev[i] + (boundary source)`. "
            "If you build that diagonal and THEN also add `condL * (U_prev[i-1] - "
            "U_prev[i])` and `condR * (U_prev[i+1] - U_prev[i])` to the right-hand "
            "side, you have counted the interior fluxes twice, once implicitly on "
            "the left and once explicitly on the right. Measured on this project: a "
            "generated solver did exactly that in its moisture block, and its "
            "moisture field never moved from its initial value across the whole "
            "run, so every subproblem failed.",
            "- Measure the SURFACE conductance on the surface area, and check its "
            "ratio to the interior conductance. The one dimensionless number that "
            "decides how fast a surface exchanges with its surroundings is the Biot "
            "number, and you can compute it from the problem's own constants before "
            "you run anything: `Bi = h*R/k` for heat, and the same shape with the "
            "mass-transfer coefficient and the diffusivity for moisture. Print the "
            "value your code is effectively using. If you rescale the radius to "
            "`xi = r/R`, the conduction conductance and the convection conductance "
            "must both be expressed in that same measure — a boundary term carrying "
            "an extra or missing factor of `R` still runs, still converges, and "
            "silently changes Bi by `R**2`. Measured on this project: a generated "
            "solver's moisture block annotated its surface flux as "
            "`R * km * (C - Cair)` and its diffusivity as `rho_d * D / R**2`, a "
            "combination whose ratio to the interior conductance is off by a factor "
            "of `R**2`; the run then integrated ten days while the moisture stayed "
            "essentially at its initial value.",
            "- Sanity-check your answer against the duration the problem states, not "
            "just against your own code. If the statement says the process takes a "
            "couple of days, then a simulation that runs for ten days and leaves the "
            "material almost unchanged is reporting that your transport is wrong by "
            "orders of magnitude — that is a defect in your model, not a property of "
            "the problem. Likewise, if the answer arrives almost instantly, your "
            "transport is too fast. State the drying or settling time you compute "
            "and compare it with the timescale the statement gives.",
            "- BUT DO NOT DISTORT YOUR TRANSPORT TO MATCH A STATED DURATION. The "
            "comparison above catches stalls and instant answers; it is NOT a "
            "licence to bend the physics. A sentence like `the process generally "
            "lasts 2-3 days` is DESCRIPTIVE PROSE about the usual case, not the "
            "value the problem asks you to find, and it is not in the official "
            "template. Measured on this project: a rule telling the solver to "
            "`fix the model` whenever its computed time disagreed with that "
            "sentence produced runs that dried the material 19x, 50x and 66x too "
            "fast -- every one of them physically wrong, and one arriving at a "
            "moisture of 0.033 kg/kg after only 1800 s. An earlier version of "
            "this note backed the principle with a 68-69 day drying time; that "
            "figure was WRONG -- it came from problem 1's diffusivity -- so it is "
            "gone and you should not reproduce it. The principle itself stands "
            "without it. When your computed duration disagrees with a "
            "descriptive sentence, the sentence is what yields: keep your "
            "transport, and report the duration your model actually gives. "
            "Distorting the physics to match prose is the more serious error of "
            "the two. But note the converse, which the deleted figure used to "
            "obscure: a duration that disagrees with the sentence is often a "
            "sign your transport is too slow, so check the diffusivity against "
            "the appendix the problem specifies before concluding the sentence "
            "is the one at fault.",
            "- Test the scheme's STEADY STATE as well as its transient, and report "
            "the error as `steady_state_error`. Run the scheme far past the "
            "transient and compare with the exact steady solution of the same "
            "problem (for convection at the surface, the uniform ambient value). "
            "This is the cheapest way to catch a flux-bookkeeping error, because at "
            "convergence `U_new = U_prev = U_ss` must satisfy the discrete steady "
            "equation: substitute a constant field into your own update and check "
            "that it stays constant. A double-counted flux or a misplaced "
            "control-volume factor violates that identity even when the t=0 check "
            "passes, and a scheme that cannot hold a steady state cannot produce a "
            "correct transient either.",
            "- Validate the scheme before trusting it, on a case that is completely "
            "self-contained. Run the same scheme on a problem whose answer is known "
            "exactly (constant coefficients with a simple boundary condition, an "
            "analytical series, or a second independent method) and measure the "
            "maximum absolute deviation. The validation case must reproduce EVERY "
            "assumption of the exact solution it is compared against: the same "
            "geometry, the same coefficients, and above all the same boundary "
            "value. Driving the validation boundary with the real problem's air "
            "temperature while comparing against a series that assumes a different "
            "ambient measures the gap between two different problems, not your "
            "scheme's error, and the number it reports is worthless. Also re-run "
            "the validation at dt and at dt/2: a scheme whose answer ignores the "
            "time step has a broken storage term. Fix the scheme until the "
            "deviation is within tolerance. A scheme that fails its own validation "
            "must not be used to produce the answer.",
            "- Validate your REFERENCE before you validate against it. An "
            "analytical series is code too, and it is wrong more often than the "
            "scheme is. Check that the reference reproduces the conditions it "
            "claims before you compare anything to it: evaluate it at t=0 and "
            "confirm it returns the initial profile you started the scheme from, "
            "and evaluate it at large t and confirm it returns the steady state "
            "your boundary conditions imply. A reference that fails either check "
            "is the thing to fix. Its eigenvalues are the usual culprit: they must "
            "satisfy the boundary condition in the SAME units as the geometry. For "
            "a cylinder with convection at r=R the condition is "
            "beta*R*J1(beta*R) = (h*R/k)*J0(beta*R); writing "
            "beta*J1(beta*R) = (h*R/k)*J0(beta*R) instead is off by a factor of R "
            "and produces a series that matches nothing.",
            "- Predict the TIMESCALE before you integrate, and print it. From the "
            "material's diffusivity and the body's size you can compute the "
            "characteristic diffusion time `R**2 / D` and the surface-limited time "
            "`R / h_m` scaled by the driving difference, entirely from the "
            "statement's own constants, before writing a single time step. Do this "
            "first: it tells you what answer is even possible, and it catches a "
            "misread constant immediately, because a wrong diffusivity changes the "
            "timescale by orders of magnitude. Then compare the drying time your "
            "simulation actually produces against that estimate. Measured on this "
            "project: a generated solver integrated far past the process duration "
            "and reported a final moisture of 2.5217 against an initial 2.55 -- the "
            "material essentially had not dried -- which is invisible if you only "
            "check that the code runs, and obvious the moment you compare the "
            "computed time against `R**2 / D`. If your simulated drying takes "
            "orders of magnitude longer than the characteristic time, the interior "
            "transport is switched off; if it is orders of magnitude shorter, the "
            "transport is far too fast.",
            "- If you cannot derive a manufactured source term you trust, USE THIS "
            "ONE. Deriving the source term is the step that keeps going wrong: on "
            "this project a solver reported orders of -1.49, -1.90, -1.96 and "
            "-1.96 across four runs, all from a manufactured test whose source "
            "term could not be right, while the same solver's steady-state and "
            "deviation checks passed and its answer was correct. For a "
            "CONSTANT-COEFFICIENT radial diffusion validation case "
            "`du/dt = D * (1/r) * d/dr(r * du/dr) + f`, take the trial field "
            "`u(r, t) = (1 + r**2 / R**2) * (1 + t / tau)` and then, term by term: "
            "`du/dt = (1 + r**2 / R**2) / tau` and "
            "`(1/r) * d/dr(r * du/dr) = (4 / R**2) * (1 + t / tau)`, so "
            "`f = (1 + r**2 / R**2) / tau - 4 * D * (1 + t / tau) / R**2`. "
            "Check it yourself by substituting back before you use it. This field "
            "is exact, exercises the radial operator rather than only the time "
            "march, satisfies a time-dependent Dirichlet condition at `r = R` "
            "(`u(R, t) = 2 * (1 + t / tau)`), and needs no special functions. "
            "Refine space and time together, compare at the SAME physical time at "
            "every level, and you should recover the order your scheme claims.",
            "- Declare `expected_order` as the order your scheme can ACTUALLY "
            "achieve, not the order you would like. If you refine time and space "
            "together -- halving both -- the composite order is the MINIMUM of the "
            "two, so backward Euler in time with a second-order space operator is "
            "FIRST order overall, and asserting 2 against it is asserting something "
            "your own scheme cannot deliver. Measured on this project: a solver "
            "whose manufactured test finally measured a healthy order of 0.991 was "
            "still reported as failing, purely because it had declared 2 and "
            "compared 0.991 against that. The measurement was right and the "
            "expectation was wrong. So state how you refine, and declare the order "
            "that refinement can produce: `dt` and `dx` scaled together gives "
            "`min(order_t, order_x)`; refining only space gives the space order "
            "alone; refining only time gives the time order alone.",
            "- Validate your scheme in STAGES, simplest first, and report which "
            "stage you reached. Build and converge the CONSTANT-coefficient version "
            "before you add variable coefficients, coupling, or an iterative "
            "nonlinear solve. A scheme that cannot hold a constant-coefficient "
            "solution will not hold a coupled variable-coefficient one, and when "
            "you add everything at once you cannot tell which addition broke it. "
            "Measured on this project: solvers that validated a "
            "constant-coefficient case first passed it with an error of 0.0148, "
            "while solvers that went straight to a coupled variable-coefficient "
            "Picard scheme reported a convergence order of -1.49 or had a field "
            "that never left its initial value, on every one of many attempts. "
            "Report the validation numbers for the constant-coefficient stage "
            "SEPARATELY from the full model, so a repair knows which stage is "
            "broken and does not rewrite the stage that already works.",
            "- Prefer a MANUFACTURED solution over a special-function reference "
            "wherever you can, because it removes the whole class of errors above. "
            "Pick a smooth trial field, substitute it into your governing equation "
            "by hand to obtain the source term it implies, add that source to your "
            "scheme, and then refine the grid: the error between your scheme and the "
            "manufactured field must fall at the order your discretisation claims. "
            "This needs no Bessel functions, no eigenvalues and no series — only "
            "arithmetic you can check — and it measures the SCHEME, which is what "
            "you are trying to validate. Report `manufactured_order` (the order you "
            "observe from two or three grid levels) and `expected_order` (the order "
            "your scheme claims, 1 for backward Euler in time and normally 2 for a "
            "centred spatial discretisation). A scheme that loses a control-volume "
            "factor or counts a flux twice does not show the expected order, so this "
            "test catches exactly the faults a wrong reference hides. Measured on "
            "this project: analytic references were the broken part in several "
            "separate runs, each time blocking a scheme that was sound.",
            "- AUDIT CONSERVATION and report the audit as a number. A spurious or "
            "missing density factor in ONE term of a balance does not look wrong "
            "locally -- the code runs, the curves are smooth, the scheme "
            "converges -- so you cannot find it by reading your own output. You "
            "find it by checking that the conserved quantity actually adds up. "
            "Measured on this project: a generated solver wrote its moisture "
            "storage as `V/dt` (no density) but its surface flux as "
            "`rho * H_MASS * Ab` (with density), so surface exchange was inflated "
            "by `rho ~ 820` and the material dried from 2.55 to below 0.15 kg/kg "
            "in under 180 s against a process the statement says takes 2-3 days. "
            "Every term of ONE balance must carry the SAME factors: if the storage "
            "term contains the density, the flux terms must too; if it does not, "
            "they must not either. Take the density out and the width factor too, "
            "and check what is left is dimensionally consistent. So compute over "
            "the whole simulation `mass_balance_relative_error = |stored change - "
            "integral of boundary flux dt| / (|stored change| + scale)` and report "
            "it; do the same for heat as `energy_balance_relative_error`. A correct "
            "discretisation drives these toward zero as you refine; a factor error "
            "leaves them O(1) and names the field that is wrong.",
            "- Report that validation in the final JSON summary as a `self_check` "
            "object:",
            '   "self_check": {"case": "<what you compared against>", '
            '"max_abs_error": <float>, "initial_profile_error": <float>, '
            '"steady_state_error": <float>, '
            '"mass_balance_relative_error": <float>, '
            '"energy_balance_relative_error": <float>, '
            '"profile_span": <float>, "reference_span": <float>, '
            '"profile_min": <float>, "profile_max": <float>, '
            '"initial_value": <float>, "boundary_value": <float>, '
            '"tolerance": <float>, "passed": <bool>, '
            '"failed_criteria": [<names>], '
            '"manufactured_order": <float>, "expected_order": <float>, '
            '"manufactured_levels": [{"n": <int>, "cells": <int>, "error": <float>}]}',
            "   `failed_criteria` must name the criteria that actually failed, "
            "using these names: \"max_abs_error\", \"initial_profile_error\", "
            "\"steady_state_error\", \"profile_span\", \"dt_sensitivity\", "
            "\"manufactured_order\". Leave it "
            "`[]` when nothing failed. Your `passed` verdict must agree with your own "
            "numbers: a `passed=false` alongside measurements that are all inside the "
            "tolerance you stated is a contradiction a reader cannot resolve, and it "
            "will be reported as exactly that rather than acted on. Measured on this "
            "project: a solver reported max_abs_error=0.0148 against a stated "
            "tolerance of 1.0, a steady-state error of 4.3e-11, and a profile span of "
            "14.149 against its reference's 14.173 — its scheme was validated by its "
            "own evidence — and then set `passed=false` with no explanation, which "
            "blocked the run. Say which criterion failed, or set `passed` to match "
            "the measurements.",
            "   `initial_profile_error` is the deviation at t=0 between your "
            "scheme's profile and your reference's profile for that same case — "
            "measure it and print it, do not reason about it. It is the cheapest "
            "test you have: your scheme starts from the initial condition you gave "
            "it, so a reference that disagrees with it at t=0 is a broken "
            "reference, and you must fix the reference rather than the scheme. "
            "Measuring this first is what stops you from chasing a scheme that was "
            "right all along.",
            "   `steady_state_error` is the deviation between your scheme, run far "
            "past the transient, and the exact steady state of the same case — "
            "again measured, not reasoned about. Together these two bracket the "
            "answer: t=0 tests the reference, steady state tests the scheme's flux "
            "bookkeeping. A scheme whose t=0 error is tiny but whose steady-state "
            "error is large has a structural fault (a flux counted twice, a "
            "control-volume factor misplaced), and no amount of time-step "
            "refinement will repair it.",
            "   `profile_min` and `profile_max` are the smallest and largest values "
            "of YOUR SCHEME's profile at the validation time, and `initial_value` "
            "and `boundary_value` are the uniform value you started from and the "
            "value the boundary condition drives the field toward. Print all four. "
            "A flat profile has two very different causes and only these numbers "
            "separate them: if your flat profile sits at `boundary_value`, your "
            "interior conductance is too large; if it still sits at `initial_value`, "
            "your surface condition is not acting at all and the fault is in the "
            "boundary row, not the conductance. Measured on this project: a solver "
            "reported a span of 8.5e-14 while its reference spanned 10.58, and "
            "without these values the fault could not be located from the record.",
            "   `profile_span` is max minus min of YOUR SCHEME's profile across the "
            "radius at the validation time, and `reference_span` is the same "
            "quantity for the REFERENCE. Print both. They diagnose the single most "
            "common way a diffusion scheme is wrong: if your profile is nearly "
            "FLAT where the reference is strongly curved, your diffusion "
            "conductance is too large. That means a stray geometric factor in the "
            "conduction term — an extra `1/R**2`, or a control-volume width "
            "missing from the storage term — not a sign error and not a "
            "time-stepping problem. A conductance that is too large equilibrates "
            "the profile almost immediately, so the centre and the surface come "
            "out nearly equal and the error looks like a uniform offset. Fix the "
            "factor; do not shrink the time step.",
            "   Set `passed` to false if you could not reach the tolerance. Never "
            "report `passed: true` unless the measured deviation really is within "
            "the tolerance you state.",
            "",
            "## NAMED OPERATORS (hard rule)",
            "Expose your discretisation as importable functions with these EXACT "
            "names, each taking explicit arguments and returning arrays, with no "
            "file I/O and no printing inside them:",
            "- `assemble_fixed(grid, properties, dt) -> (matrix, rhs_base)` — builds "
            "the linear system for one implicit step.",
            "- `step_fixed(matrix, rhs) -> solution` — solves that system.",
            "This is a hard requirement with a concrete reason. A solver on this "
            "project was blocked by its own convergence test reporting order "
            "approximately ZERO: the error did not fall as the grid was refined. "
            "That single symptom is consistent with TWO different faults -- a wrong "
            "source term in the test, or a wrong DISCRETISATION OPERATOR such as a "
            "diffusion term missing its control-volume width factor -- and nothing "
            "the program reported could separate them, because both are evaluated "
            "through the same self-authored test. With the operators named and "
            "importable, an INDEPENDENT reviewer can drive them on a case whose "
            "exact solution is known and tell the two apart. A wrong source term "
            "leaves the operator accurate; a missing control-volume factor does "
            "not. Keep the functions pure so that they can be called this way, and "
            "put the model-specific setup in `main()`, not inside them.",
            "## GRID AND OUTPUT BOOKKEEPING (hard rules)",
            "- COPY THE TEMPLATE'S HEADER ROW CELL FOR CELL, INCLUDING TEXT "
            "LABELS. Do not assume every header cell is a number. Measured on "
            "this project: a run wrote the last column of result4.xlsx as the "
            "numeric distance 2.0, while the official template's last header "
            "cell is the TEXT label 药材表面 (the herb surface). The file was "
            "otherwise well formed, and the run stayed blocked on that single "
            "header cell through six solve attempts and three verifications, "
            "because every repair changed something other than that cell. Where "
            "the template labels a column in words, write exactly those words. "
            "AND STOP STRICTLY PAST THE TARGET, NOT ON IT. The problem requires "
            "the final moisture to be strictly BELOW 0.15 kg/kg. A run ended its "
            "grid at t=149340 s on a final row whose largest value was exactly "
            "0.150000, and was failed for it -- the drying time was right, the "
            "stopping rule was not. Never end the grid on the row where the "
            "value equals the threshold: keep integrating until every point is "
            "STRICTLY less, and report the first time at which that holds. "
            "AND VALIDATE THE REFERENCE BEFORE YOU TRUST THE SELF-CHECK. A "
            "run's scheme validation reported passed=false against an "
            "analytical Bessel-series reference, and the reviewer's conclusion "
            "was that the REFERENCE, not the scheme, was the more likely fault: "
            "a reference that cannot reproduce the initial profile condemns the "
            "scheme for the reference's own error. So when you build a "
            "self-check, first measure the deviation between your scheme and "
            "your reference at t=0 and report it as `initial_profile_error`; if "
            "the reference does not reproduce the initial profile, fix the "
            "REFERENCE and say so, rather than rewriting a scheme that was never "
            "shown to be wrong. AND MATCH THE SCHEME TO THE ORDER YOU CLAIM. A run shipped a "
            "manufactured-solution self-check declaring expected_order = 2.0 "
            "while its time integration was first-order, so the check reported "
            "an order below the expectation and failed -- the generator then "
            "could not repair its way out, because the shortfall was inherent in "
            "the scheme it had chosen. Either use a time discretisation that is "
            "genuinely second order (Crank-Nicolson, or BDF2 with a consistent "
            "start) so a second-order claim is true in BOTH space and time, or "
            "declare the expectation your scheme actually delivers -- but never "
            "lower the declared expectation merely to make the test pass, and "
            "never leave the two inconsistent. "
            "threshold is a REPORTING CRITERION, not a physical bound: do NOT "
            "clamp, floor, saturate or overwrite any moisture value with it. One "
            "run's surface column fell from 0.1501 to exactly 0.15 and then "
            "repeated 0.15 for every remaining row (stored as the exact double "
            "0.14999999999999999, i.e. the literal threshold was written, not a "
            "computed value), so it could never satisfy a strict inequality. Let "
            "the computed value stand, even when it crosses below the target. "
            "AND THE ROW TIMES ARE A FIXED GRID TOO. The problem prescribes the "
            "interval between rows: result1 and result2 carry a row every 1 s, "
            "result3 and result4 every 60 s. A run emitted the wrong interval for "
            "result3 and was failed by the checker for it. Emit rows ON that "
            "interval, from t = 0, and do not substitute your own time step. "
            "AND STORE EACH HEADER CELL IN THE TYPE THE TEMPLATE USES. The "
            "template's distance headers are NUMBERS (0, 0.1, 0.2, ...), not text. "
            "A run wrote them as the strings '0.1' and '0.2', which reads correctly "
            "to the eye but is a different cell type from the template's. Write "
            "numeric headers as numbers and text headers as text. "
            "AND THE DISTANCE COLUMNS THEMSELVES ARE NOT YOURS TO CHOOSE. The "
            "template's data columns are a FIXED NUMERIC GRID: 0, 0.1, 0.2, ... out "
            "to the outer radius, exactly as the template file writes them. A run "
            "emitted 0.1052... in column 3 where the template has 0.1, and failed "
            "on that alone. Evaluate your model ON the template's grid; if your "
            "mesh is finer, INTERPOLATE onto the template's columns rather than "
            "writing a grid of your own. "
            "WRITE THEM IN PLACE OF THE VALUE THAT WOULD OTHERWISE SIT THERE -- "
            "do not append an extra column. A follow-up run obeyed this rule by "
            "keeping the numeric 2.0 AND adding a 药材表面 column after it, which "
            "gave result4.xlsx 23 columns where the template has 22: one mismatch "
            "was traded for another. Your output file must have EXACTLY as many "
            "columns as the template file, in the same order, so a label always "
            "REPLACES the header cell it belongs to and never adds one. If a "
            "label names a physical location that is also the outer radius, that "
            "column is the label and the radius column before it carries the "
            "numbers.",
            "- RUN THE PROCESS UNTIL THE STATED TARGET IS MET, and let the "
            "duration that requires be your answer. Where a problem asks you to "
            "determine the time a process needs, the duration IS the answer; do "
            "not pick a window in advance and stop there. Measured on this "
            "project: three separate runs all integrated exactly 259200 s (72 h) "
            "because the statement says the process `generally lasts 2-3 days`, "
            "then reported final moisture of 0.197, 0.2496 and 2.55 -- none of "
            "which met the required 0.15 kg/kg. That sentence is descriptive "
            "prose, it is NOT the problem's requirement, and the official "
            "template contains no duration limit at all (its time column reads "
            "60, 120, 180, followed by an ellipsis). Integrate to a horizon that is "
            "generous for the physics you actually coded, and if the target is "
            "still unmet at the end, extend and integrate again rather than "
            "reporting a final row that misses it. Use whichever appendix the "
            "problem under study specifies: problems 2 and 3 use the diffusivity "
            "D = 2.4e-3*exp(-0.45/C)*exp(-3850/T) with T in Kelvin, which is far "
            "larger at low moisture than problem 1's 7e-9*exp(-0.89/C). A "
            "previous version of this note cited a 68-69 day drying time; that "
            "figure was WRONG -- it was computed with problem 1's diffusivity -- "
            "and a solver that trusts it will integrate for months and still "
            "report drying far slower than the physics allows.",
            "- AFTER WRITING EACH FILE, REOPEN IT AND CHECK IT AGAINST THE "
            "REQUIREMENT. Your intent is not evidence; the file on disk is. "
            "Measured on this project: a solver's own header comment said it would "
            "`run for problems 1-4 with the sub-second/60-s output grids the "
            "template requires`, while the files it actually wrote stepped by 60 s "
            "where 1 s was required and 3600 s where 60 s was required -- and it "
            "was blocked on that for four consecutive repair attempts, each of "
            "which changed the code and the comment without changing the behaviour. "
            "It never noticed, because it checked what it meant to do rather than "
            "what it produced. So make the program read its own artifacts back: "
            "reopen every file it wrote, parse the time column, and assert the "
            "actual step equals the step the problem requires; assert the header "
            "length equals every row length; assert each required worksheet is "
            "present under the name the template uses; and assert the final row "
            "satisfies any numeric target the problem states. A mismatch must "
            "print a specific message and exit non-zero. A silent wrong file is "
            "worse than a crash, because a crash is cheap to repair and a wrong "
            "file is only found later.",
            "These rules exist because of a real, repeated failure on this project: "
            "a generated solver failed all six of its repair attempts, each time "
            "with a different symptom of ONE unresolved off-by-one. The six errors "
            "were `broadcast shapes (40,) (41,)`, `22 columns passed, passed data "
            "had 21 columns`, `header/data mismatch: hdr=22 T=21 C=21`, `fp and xp "
            "are not of the same length`, and `grid size 21 != profile length 20`. "
            "Every one is the same bug: the number of grid nodes and the length of "
            "a profile disagreed. Each repair patched the call site the error "
            "happened to surface at, so the error simply moved to the next site.",
            "- Fix the convention ONCE, at its source, and derive everything from "
            "it. Define the node count in exactly one place (for example "
            "`N_NODES = 41` meaning nodes at `xi = linspace(0, 1, N_NODES)`) and "
            "derive every array length, every interpolation grid, and every header "
            "from that single constant. Never let a literal like `20`, `21`, `40` "
            "or `41` appear anywhere else: a second copy of the node count is what "
            "drifts out of sync.",
            "- Do not confuse the INTERNAL grid with the OUTPUT columns. The output "
            "template fixes the radial sampling positions (for example 0, 0.1, ..., "
            "2.0 cm, giving 21 radial columns plus one time column = 22 columns). "
            "The number of internal nodes is YOUR choice and must never determine "
            "the output shape. Sample the internal solution onto the template's "
            "fixed positions with an interpolation whose two arrays you have "
            "asserted are the same length first.",
            "- Prove the WHOLE pipeline on a TINY case before the real run. "
            "Measured on this project across 39 runs and 162 attempts: 94 attempts "
            "failed, and 61 of those (65%) died on a pure code fault -- "
            "`IndexError`, `ValueError: inhomogeneous shape`, `operands could not "
            "be broadcast` -- before reaching any physics at all. Only 24% were the "
            "solver's own physics assertion. So most of every repair budget is "
            "spent on code faults rather than on modelling, and the physics gets "
            "roughly one attempt in three. The cheap fix is to make those faults "
            "impossible to miss: after defining your functions, run the COMPLETE "
            "sequence -- build the grid, assemble, step a few times, sample the "
            "profile, build every output row and header -- on a deliberately tiny "
            "case (3 time steps, your normal node count) inside a `try`, and print "
            "the shapes you produced before doing any real work. A shape or "
            "interpolation fault then surfaces in about a second instead of after "
            "a minutes-long integration, and you will see it yourself rather than "
            "spending the attempt. Assert `len(row) == len(header)` at the moment "
            "you append each row, so the failure names the row rather than the "
            "line that later consumed it.",
            "- When the same class of error appears a second time, your repair is "
            "wrong. Patching the site where the error surfaced cannot help, because "
            "the disagreement is between two definitions elsewhere. Stop, find the "
            "single place that defines each length, and make them agree. A repair "
            "that makes the error move to a different line has not fixed anything.",
            "",
            "## RUNTIME BUDGET (hard limit)",
            f"- Your program is killed after {PROGRAM_TIME_BUDGET_SECONDS} seconds. It "
            "must finish the WHOLE problem — every subproblem and every output "
            "file — well inside that. A program that runs out of time produces no "
            "results at all, so an accurate scheme that cannot finish is worth "
            "less than a slightly coarser one that can.",
            "- Budget for it deliberately. Choose the radial cell count and the "
            "time step so the total number of implicit solves fits the budget, and "
            "re-check that product before you commit to a scheme. Prefer a moderate "
            "grid with a vectorised update over a fine grid driven by a Python loop "
            "over time steps, and do not add iterations per step you cannot afford "
            "at every step.",
            f"- You also have {PROGRAM_MEMORY_BUDGET_MB} MB of memory, and exceeding it "
            "kills the program just as dead as running out of time. Measured on this "
            "project: two of three attempts were killed for memory after about "
            "160 seconds while a third finished the same problem in 14 seconds, "
            "because the model vectorised over BOTH space and time and "
            "materialised the whole history. Keep the working set to the current "
            "and previous fields plus the rows you must write out. Do not build a "
            "`(n_steps, n_nodes)` array, do not accumulate a list of full profiles "
            "when you only need the rows you will output, and do not allocate a "
            "dense matrix where a tridiagonal solve will do. A program killed for "
            "memory produces no results at all.",
            "- Print progress and flush it, so a program that is stopped leaves "
            "evidence of how far it got. Print one line when each subproblem "
            "STARTS and one when it FINISHES, for example "
            '`print(f"problem 3 done in {elapsed:.1f}s, records={n}", flush=True)`.',
            "- If you also print inside a time loop, guard it with a plain integer "
            "counter you increment yourself (`steps += 1`). Never guard it with a "
            "comparison against a variable that might be an array: `if k % 500 == 0:` "
            "raises `ValueError: The truth value of an array with more than one "
            "element is ambiguous` whenever `k` is a numpy array, which kills the "
            "whole program and throws away every result already computed. That "
            "exact line has already destroyed a run that had finished three of four "
            "subproblems correctly.",
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
            "- When the problem asks for a table sampled at a fixed interval, EVERY "
            "row must sit on that grid: the step between consecutive time rows has "
            "to equal the stated interval from the first row to the last. Do not "
            "append an extra row for an event such as the drying time — one "
            "off-grid row invalidates the whole table. Report a derived time such "
            "as the drying time in the JSON summary instead.",
            "- When the problem states a threshold the answer must satisfy (for "
            "example a final moisture concentration strictly below 0.15 kg/kg), "
            "satisfy it STRICTLY. A final row whose value equals the threshold is a "
            "failure, not a pass. Keep integrating until the value is below it, and "
            "assert the strict inequality on the final row before writing the file.",
            "- Leave MARGIN, and test the value you actually WRITE. You will round "
            "your output columns to a few decimals, and the check reads the file, "
            "not your unrounded array. A true value of 0.14996 rounded to four "
            "decimals is written as 0.1500, which is not strictly below 0.15, so a "
            "compliant computation still fails. Measured on this project: a run "
            "reported a final row of exactly 0.150000 for that reason. Integrate "
            "past the crossing until the ROUNDED final value is comfortably below "
            "the threshold (a few percent of margin, not a hair), and assert the "
            "strict inequality on the rounded value you are about to write.",
            "- Do not stop the integration at the exact instant the threshold is "
            "crossed either. Your time grid is fixed, so the last row you write "
            "should be at a time where the value has clearly passed the threshold, "
            "not sitting on it.",
            "- When the model integrates a differential equation, `self_check` is "
            "MANDATORY and must be present even if it failed. Omitting it is a "
            "defect: it leaves the scheme unvalidated and the run is rejected.",
        ])

    @staticmethod
    def _repair_block(previous_code: str, failure_feedback: list[str]) -> str:
        truncated = len(previous_code) > REPAIR_CODE_MAX_CHARS
        excerpt = previous_code[:REPAIR_CODE_MAX_CHARS]
        return "\n".join([
            "",
            "## PREVIOUS ATTEMPT FAILED — REPAIR IT",
            "The previous program did not satisfy the requirements. Issues found:",
            *[f"  - {item}" for item in failure_feedback],
            "",
            "Previous program (the complete source — fix the actual cause):",
            "```python",
            excerpt,
            "```",
            *(
                [
                    "",
                    "NOTE: that excerpt was cut off for length, so the remainder of "
                    "your previous program is not shown. Re-derive the missing part "
                    "rather than dropping it.",
                ]
                if truncated
                else []
            ),
            "",
            "Return the corrected COMPLETE program as raw Python source only "
            "(no markdown fences, no commentary). Fix the underlying computation or "
            "data handling rather than merely suppressing the error, and change ONLY "
            "what the reported issues require: every part the issues do not mention "
            "already works, so keep it exactly as it was. A repair that fixes the "
            "listed issue while changing something else has made the program worse, "
            "not better.",
        ])


class SandboxProgramRunner:
    """Runs a generated program in the Docker sandbox and keeps its artifacts."""

    def __init__(
        self,
        image: str = DEFAULT_SANDBOX_IMAGE,
        timeout_seconds: int = PROGRAM_TIME_BUDGET_SECONDS,
        max_memory_mb: int = PROGRAM_MEMORY_BUDGET_MB,
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
    """Extract the last JSON object the program printed.

    The contract asks for that object on its own line, but a program that
    pretty-prints it across several lines is still reporting its results.
    Reading only whole single lines turned a complete answer into "no
    statistics", which is indistinguishable from the solver having produced
    nothing — and it blamed the solver for the host's own reader.
    """
    lines = stdout.splitlines(keepends=True)
    offsets: list[int] = []
    running = 0
    for line in lines:
        stripped = line.lstrip()
        offsets.append(running + (len(line) - len(stripped)))
        running += len(line)

    # The summary is printed last, so scan backwards and stop at the first
    # object that parses. The cap keeps a pathological stdout from turning this
    # into a quadratic scan.
    decoder = json.JSONDecoder()
    candidates = [
        offset
        for line, offset in zip(lines, offsets)
        if line.lstrip().startswith("{")
    ]
    for offset in reversed(candidates[-200:]):
        try:
            parsed, _ = decoder.raw_decode(stdout, offset)
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
