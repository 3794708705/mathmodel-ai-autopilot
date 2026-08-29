"""MathModel AI — Model compiler, solver abstraction, and constraint evaluator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from mathmodel.domain.math_model import MathematicalModel, ConstraintRelation, ObjectiveSense

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SolverResult
# ═══════════════════════════════════════════════════════════════

class SolverStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class SolverResult:
    """Result of a solver execution."""

    solver_run_id: str = field(default_factory=lambda: f"SOLVER-{uuid4().hex[:8]}")
    model_id: str = ""
    solver: str = ""
    solver_version: str = ""
    status: SolverStatus = SolverStatus.UNKNOWN
    objective_value: Optional[float] = None
    variable_values: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    iterations: int = 0
    gap: Optional[float] = None
    feasibility: bool = False
    warnings: list[str] = field(default_factory=list)
    raw_status: str = ""
    execution_record_id: str = ""
    execution_real: bool = False
    production_safe: bool = False


# ═══════════════════════════════════════════════════════════════
# Base Solver Adapter
# ═══════════════════════════════════════════════════════════════

class BaseSolverAdapter:
    """Abstract interface for solver adapters."""

    @property
    def solver_name(self) -> str:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        raise NotImplementedError

    def solve(self, model: MathematicalModel, **kwargs) -> SolverResult:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════
# SciPy Solver Adapter
# ═══════════════════════════════════════════════════════════════

class SciPySolverAdapter(BaseSolverAdapter):
    """Solver adapter using SciPy's linprog for LP problems."""

    @property
    def solver_name(self) -> str:
        return "scipy"

    @property
    def available(self) -> bool:
        try:
            import scipy.optimize
            return True
        except ImportError:
            return False

    def solve(self, model: MathematicalModel, **kwargs) -> SolverResult:
        """Solve an LP model using SciPy linprog via the compiler."""
        if not self.available:
            return SolverResult(
                model_id=model.model_id,
                solver="scipy",
                status=SolverStatus.ERROR,
                warnings=["SciPy not available"],
            )

        # Compile the model to LP form
        compiled = SimpleLPCompiler.compile(model)
        compiled["model_id"] = model.model_id

        # Delegate to the real solver function
        return solve_lp_scipy(compiled)


# ═══════════════════════════════════════════════════════════════
# Simple LP Compiler
# ═══════════════════════════════════════════════════════════════

class SimpleLPCompiler:
    """Compiles a MathematicalModel into a canonical LP form.

    Produces: c (objective coefficients), A_ub, b_ub, A_eq, b_eq, bounds.
    """

    @staticmethod
    def compile(model: MathematicalModel) -> dict[str, Any]:
        """Compile model to LP components."""
        var_map = {v.symbol: i for i, v in enumerate(model.variables)}
        n = len(model.variables)

        # Objective: assume first objective, negate for maximization
        c = [0.0] * n
        if model.objectives:
            obj = model.objectives[0]
            c = SimpleLPCompiler._parse_coefficients(obj.expression, var_map, n)
            # linprog MINIMIZES — negate for maximization
            if obj.sense == ObjectiveSense.MAXIMIZE:
                c = [-x for x in c]

        # Constraints
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        bounds = [(0.0, None)] * n  # Default non-negative

        for con in model.constraints:
            coeffs = SimpleLPCompiler._parse_coefficients(con.expression, var_map, n)
            if con.relation == ConstraintRelation.LE:
                A_ub.append(coeffs)
                b_ub.append(con.rhs)
            elif con.relation == ConstraintRelation.GE:
                A_ub.append([-x for x in coeffs])
                b_ub.append(-con.rhs)
            elif con.relation == ConstraintRelation.EQ:
                A_eq.append(coeffs)
                b_eq.append(con.rhs)

        # Apply variable bounds
        for i, v in enumerate(model.variables):
            lb = v.lower_bound if v.lower_bound is not None else 0.0
            ub = v.upper_bound
            bounds[i] = (lb, ub)

        return {
            "c": c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": bounds,
            "variable_names": [v.symbol for v in model.variables],
            "maximize": model.objectives[0].sense == ObjectiveSense.MAXIMIZE if model.objectives else False,
        }

    @staticmethod
    def _parse_coefficients(expr: str, var_map: dict[str, int], n: int) -> list[float]:
        """Parse a simple linear expression like '2*x1 + 1.5*x2 + x3'."""
        import re
        coeffs = [0.0] * n

        # Remove whitespace
        expr = expr.replace(" ", "")

        # Split by + and -
        terms = re.split(r'(?=[+-])', expr)
        for term in terms:
            if not term:
                continue
            # Match coefficient * variable
            match = re.match(r'([+-]?\d*\.?\d*)\*?(\w+)', term)
            if match:
                coeff_str = match.group(1)
                var_name = match.group(2)

                if coeff_str in ("", "+", "-"):
                    coeff = 1.0 if coeff_str != "-" else -1.0
                else:
                    coeff = float(coeff_str)

                if var_name in var_map:
                    coeffs[var_map[var_name]] = coeff
        return coeffs


# ═══════════════════════════════════════════════════════════════
# Real SciPy LP Solver
# ═══════════════════════════════════════════════════════════════

def solve_lp_scipy(compiled: dict[str, Any]) -> SolverResult:
    """Solve a compiled LP using SciPy and return a SolverResult."""
    import time
    import numpy as np

    try:
        from scipy.optimize import linprog
    except ImportError:
        return SolverResult(solver="scipy", status=SolverStatus.ERROR, warnings=["SciPy not installed"])

    start = time.time()

    try:
        result = linprog(
            c=compiled["c"],
            A_ub=np.array(compiled["A_ub"]) if compiled["A_ub"] else None,
            b_ub=np.array(compiled["b_ub"]) if compiled["b_ub"] else None,
            A_eq=np.array(compiled["A_eq"]) if compiled["A_eq"] else None,
            b_eq=np.array(compiled["b_eq"]) if compiled["b_eq"] else None,
            bounds=compiled["bounds"],
            method="highs",
        )

        elapsed = time.time() - start

        status_map = {
            0: SolverStatus.OPTIMAL,
            1: SolverStatus.TIME_LIMIT,
            2: SolverStatus.INFEASIBLE,
            3: SolverStatus.UNBOUNDED,
        }

        var_values = {}
        if result.success and result.x is not None:
            for i, name in enumerate(compiled["variable_names"]):
                var_values[name] = float(result.x[i])

        obj_val = float(result.fun) if result.fun is not None else None
        # Negate back for maximization
        if compiled.get("maximize") and obj_val is not None:
            obj_val = -obj_val

        return SolverResult(
            model_id=compiled.get("model_id", ""),
            solver="scipy",
            solver_version="scipy.optimize.linprog",
            status=status_map.get(result.status, SolverStatus.UNKNOWN),
            objective_value=obj_val,
            variable_values=var_values,
            runtime_seconds=round(elapsed, 3),
            iterations=result.nit if hasattr(result, "nit") else 0,
            feasibility=result.success,
            raw_status=str(result.status),
            execution_real=True,
            production_safe=False,
        )

    except Exception as e:
        return SolverResult(solver="scipy", status=SolverStatus.ERROR, warnings=[str(e)])


# ═══════════════════════════════════════════════════════════════
# Constraint Evaluator
# ═══════════════════════════════════════════════════════════════

def evaluate_constraints(
    model: MathematicalModel,
    result: SolverResult,
) -> dict[str, Any]:
    """Re-evaluate all constraints against solver results.

    Uses compiled coefficients — never eval() on raw expressions.
    """
    violations = []
    var_vals = result.variable_values

    # Compile the model to get coefficient matrices
    compiled = SimpleLPCompiler.compile(model)

    for i, con in enumerate(model.constraints):
        try:
            # Get the constraint row from the compiled matrices
            # For LE constraints: row is in A_ub, b_ub
            # For GE constraints: row is negated in A_ub, b_ub
            # For EQ constraints: row is in A_eq, b_eq
            coeffs = SimpleLPCompiler._parse_coefficients(
                con.expression,
                {v.symbol: j for j, v in enumerate(model.variables)},
                len(model.variables),
            )

            # Compute lhs = sum(coeff_i * var_i)
            lhs = sum(
                coeffs[j] * var_vals.get(model.variables[j].symbol, 0.0)
                for j in range(len(model.variables))
            )

            violation = 0.0
            if con.relation == ConstraintRelation.LE:
                violation = max(0.0, lhs - con.rhs)
            elif con.relation == ConstraintRelation.GE:
                violation = max(0.0, con.rhs - lhs)
            elif con.relation == ConstraintRelation.EQ:
                violation = abs(lhs - con.rhs)

            within = violation <= con.tolerance

            violations.append({
                "constraint_id": con.constraint_id,
                "name": con.name,
                "lhs": round(lhs, 6),
                "rhs": con.rhs,
                "violation": round(violation, 10),
                "within_tolerance": within,
            })
        except Exception as e:
            violations.append({
                "constraint_id": con.constraint_id,
                "name": con.name,
                "error": str(e),
                "within_tolerance": False,
            })

    max_violation = max((v.get("violation", 0.0) for v in violations), default=0.0)
    violated = [v for v in violations if not v.get("within_tolerance", False)]

    return {
        "violations": violations,
        "max_violation": max_violation,
        "violated_constraints": [v["constraint_id"] for v in violated],
        "all_satisfied": len(violated) == 0,
    }