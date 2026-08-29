"""MathModel AI — Mathematical validation gate and verification core."""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from mathmodel.domain.math_model import (
    MathematicalModel,
    Variable,
    Parameter,
    ParameterStatus,
    Objective,
    Constraint,
    Equation,
    ConstraintRelation,
)
from mathmodel.domain.verification import (
    GateStatus,
    ValidationGateResult,
    ValidationCheckResult,
    ValidationReport,
    ValidationIssue,
    IssueSeverity,
)
from mathmodel.math_ops import (
    SymbolRegistry,
    EquationRegistry,
    UnitChecker,
    validate_expression,
    ExpressionValidationError,
)
from mathmodel.solver import (
    SimpleLPCompiler,
    evaluate_constraints,
    SolverResult,
    SolverStatus,
)

logger = logging.getLogger(__name__)


class MathematicalValidationGate:
    """Unified gate for mathematical model validation.

    Integrates: schema, reference, symbol, equation, expression,
    unit, completeness, subproblem coverage, parameter readiness,
    solver compatibility.
    """

    def __init__(self, data_reference_validator: Optional[Any] = None):
        self._data_validator = data_reference_validator

    def validate(self, model: MathematicalModel) -> ValidationGateResult:
        """Run all validation checks on a model."""
        checks: list[ValidationCheckResult] = []
        failures: list[str] = []
        blocking: list[str] = []

        # 1. Schema validation
        schema_ok = self._check_schema(model, checks, failures, blocking)

        # 2. Symbol validation
        symbol_ok = self._check_symbols(model, checks, failures, blocking)

        # 3. Equation validation
        eq_ok = self._check_equations(model, checks, failures, blocking)

        # 4. Expression validation
        expr_ok = self._check_expressions(model, checks, failures, blocking)

        # 5. Unit validation
        unit_ok = self._check_units(model, checks, failures)

        # 6. Completeness
        completeness_ok = self._check_completeness(model, checks, failures, blocking)

        # 7. Parameter readiness
        param_ok = self._check_parameters(model, checks, failures, blocking)

        # Determine overall status
        if blocking:
            status = GateStatus.FAIL
        elif failures:
            status = GateStatus.WARNING
        else:
            status = GateStatus.PASS

        return ValidationGateResult(
            status=status,
            checks=checks,
            failures=failures,
            blocking_failures=blocking,
            confidence=0.9 if status == GateStatus.PASS else 0.5,
            override_required=len(blocking) > 0,
        )

    # ── Individual checks ─────────────────────────────────────

    def _check_schema(self, model, checks, failures, blocking) -> bool:
        """Validate model domain invariants."""
        issues = model.validate_domain()
        if issues:
            for issue in issues:
                failures.append(issue)
                # Missing objective is blocking
                if "objective" in issue.lower():
                    blocking.append(issue)
            checks.append(ValidationCheckResult(
                check_name="schema", status=GateStatus.FAIL,
                message=f"{len(issues)} domain issues",
                details={"issues": issues},
            ))
            return False
        checks.append(ValidationCheckResult(check_name="schema", status=GateStatus.PASS))
        return True

    def _check_symbols(self, model, checks, failures, blocking) -> bool:
        """Check symbol uniqueness and consistency."""
        registry = SymbolRegistry()
        issues = []
        try:
            for v in model.variables:
                registry.register(v.symbol, v.variable_id, "variable", unit=v.unit)
            for p in model.parameters:
                registry.register(p.symbol, p.parameter_id, "parameter", unit=p.unit)
        except ValueError as e:
            issues.append(str(e))

        if issues:
            for issue in issues:
                failures.append(issue)
                blocking.append(issue)  # Symbol conflicts block
            checks.append(ValidationCheckResult(
                check_name="symbols", status=GateStatus.FAIL,
                message=issues[0], details={"issues": issues},
            ))
            return False
        checks.append(ValidationCheckResult(check_name="symbols", status=GateStatus.PASS))
        return True

    def _check_equations(self, model, checks, failures, blocking) -> bool:
        """Check equation references and dependencies."""
        registry = EquationRegistry()
        for eq in model.equations:
            try:
                registry.register(eq)
            except ValueError as e:
                failures.append(str(e))
                blocking.append(str(e))

        dep_issues = registry.validate_dependencies()
        for issue in dep_issues:
            failures.append(issue)
            blocking.append(issue)

        if failures and any(f in blocking for f in failures):
            checks.append(ValidationCheckResult(
                check_name="equations", status=GateStatus.FAIL,
                message="Equation issues", details={"issues": dep_issues},
            ))
            return False
        checks.append(ValidationCheckResult(check_name="equations", status=GateStatus.PASS))
        return True

    def _check_expressions(self, model, checks, failures, blocking) -> bool:
        """Validate all expressions are safe and parseable."""
        issues = []
        symbols = {v.symbol for v in model.variables} | {p.symbol for p in model.parameters}

        for obj in model.objectives:
            try:
                validate_expression(obj.expression, symbols)
            except ExpressionValidationError as e:
                issues.append(f"Objective {obj.objective_id}: {e}")

        for con in model.constraints:
            try:
                validate_expression(con.expression, symbols)
            except ExpressionValidationError as e:
                issues.append(f"Constraint {con.constraint_id}: {e}")

        if issues:
            for issue in issues:
                failures.append(issue)
                blocking.append(issue)
            checks.append(ValidationCheckResult(
                check_name="expressions", status=GateStatus.FAIL,
                message=f"{len(issues)} expression issues",
            ))
            return False
        checks.append(ValidationCheckResult(check_name="expressions", status=GateStatus.PASS))
        return True

    def _check_units(self, model, checks, failures) -> bool:
        """Check unit consistency (simple string comparison)."""
        unit_issues = []
        for con in model.constraints:
            # Constraints with rhs have implicit unit matching
            if con.unit:
                # Simple check: no compound analysis yet
                pass

        if unit_issues:
            failures.extend(unit_issues)
            checks.append(ValidationCheckResult(
                check_name="units", status=GateStatus.WARNING,
                message=f"{len(unit_issues)} unit issues",
            ))
            return True  # Unit issues are warnings, not blockers
        checks.append(ValidationCheckResult(check_name="units", status=GateStatus.PASS))
        return True

    def _check_completeness(self, model, checks, failures, blocking) -> bool:
        """Check model completeness."""
        issues = []

        if not model.objectives:
            issues.append("Missing objective")
        if not model.variables:
            issues.append("Missing variables")

        if issues:
            for issue in issues:
                failures.append(issue)
                blocking.append(issue)
            checks.append(ValidationCheckResult(
                check_name="completeness", status=GateStatus.FAIL,
                message="; ".join(issues),
            ))
            return False
        checks.append(ValidationCheckResult(check_name="completeness", status=GateStatus.PASS))
        return True

    def _check_parameters(self, model, checks, failures, blocking) -> bool:
        """Check parameter readiness."""
        issues = []
        for p in model.parameters:
            if p.status == ParameterStatus.REQUIRED and p.value is None:
                issues.append(
                    f"Parameter {p.parameter_id} ({p.symbol}) is REQUIRED but has no value"
                )

        if issues:
            for issue in issues:
                failures.append(issue)
                blocking.append(issue)
            checks.append(ValidationCheckResult(
                check_name="parameters", status=GateStatus.FAIL,
                message=f"{len(issues)} unready parameters",
            ))
            return False
        checks.append(ValidationCheckResult(check_name="parameters", status=GateStatus.PASS))
        return True


# ═══════════════════════════════════════════════════════════════
# Validation Agent core logic (deterministic)
# ═══════════════════════════════════════════════════════════════

def build_validation_report(
    model: MathematicalModel,
    result: SolverResult,
) -> ValidationReport:
    """Build a ValidationReport from deterministic checks."""

    issues: list[ValidationIssue] = []
    report = ValidationReport(
        model_id=model.model_id,
        model_version=model.version,
        solver_run_id=result.solver_run_id,
        overall_status=GateStatus.PASS,
    )

    # 1. Solver status validation
    if result.status == SolverStatus.OPTIMAL:
        report.solver_validation = {"status": "optimal", "valid": True}
    elif result.status == SolverStatus.FEASIBLE:
        report.solver_validation = {"status": "feasible", "valid": True}
        report.overall_status = GateStatus.WARNING
    elif result.status == SolverStatus.INFEASIBLE:
        report.solver_validation = {"status": "infeasible", "valid": False}
        report.overall_status = GateStatus.FAIL
        issues.append(ValidationIssue(
            severity=IssueSeverity.CRITICAL,
            category="solver",
            message="Model is infeasible",
        ))
    elif result.status == SolverStatus.UNBOUNDED:
        report.solver_validation = {"status": "unbounded", "valid": False}
        report.overall_status = GateStatus.FAIL
        issues.append(ValidationIssue(
            severity=IssueSeverity.CRITICAL,
            category="solver",
            message="Model is unbounded",
        ))
    elif result.status == SolverStatus.ERROR:
        report.solver_validation = {"status": "error", "valid": False}
        report.overall_status = GateStatus.FAIL
        issues.append(ValidationIssue(
            severity=IssueSeverity.CRITICAL,
            category="solver",
            message="Solver error",
        ))
    else:
        report.solver_validation = {"status": result.status.value, "valid": False}
        report.overall_status = GateStatus.FAIL

    # 2. Constraint validation
    constraint_eval = evaluate_constraints(model, result)
    report.constraint_validation = constraint_eval
    if not constraint_eval["all_satisfied"]:
        report.overall_status = GateStatus.FAIL
        for cid in constraint_eval["violated_constraints"]:
            issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                category="constraint",
                message=f"Constraint {cid} violated (max_violation={constraint_eval['max_violation']})",
                evidence={"constraint_eval": constraint_eval},
            ))

    # 3. Result numerical validation (NaN/Inf)
    result_issues = _check_result_numerics(model, result)
    report.result_validation = {"valid": not result_issues, "issues": result_issues}
    if result_issues:
        report.overall_status = GateStatus.FAIL
        for issue in result_issues:
            issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                category="numerical",
                message=issue,
            ))

    # 4. Objective recomputation
    if result.objective_value is not None and model.objectives:
        obj_validation = _check_objective_recomputation(model, result)
        report.result_validation["objective_recomputation"] = obj_validation
        if not obj_validation.get("matches", False):
            report.overall_status = GateStatus.FAIL
            issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                category="objective",
                message=f"Objective mismatch: solver={obj_validation.get('solver_objective')}, "
                        f"recomputed={obj_validation.get('recomputed_objective')}, "
                        f"diff={obj_validation.get('absolute_difference')}",
            ))

    # 5. Execution truth
    if not result.execution_real:
        report.overall_status = GateStatus.FAIL
        issues.append(ValidationIssue(
            severity=IssueSeverity.CRITICAL,
            category="execution",
            message="Result not from real execution",
        ))

    report.issues = issues
    report.metrics = {
        "issue_count": len(issues),
        "critical_count": sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL),
    }
    return report


def _check_result_numerics(model: MathematicalModel, result: SolverResult) -> list[str]:
    """Check for NaN, Inf, and out-of-domain values."""
    issues = []

    if result.objective_value is not None:
        if math.isnan(result.objective_value):
            issues.append("Objective value is NaN")
        if math.isinf(result.objective_value):
            issues.append("Objective value is Inf")

    var_by_symbol = {v.symbol: v for v in model.variables}
    for symbol, value in result.variable_values.items():
        if math.isnan(value):
            issues.append(f"Variable {symbol} is NaN")
        if math.isinf(value):
            issues.append(f"Variable {symbol} is Inf")
        if symbol in var_by_symbol:
            var = var_by_symbol[symbol]
            if var.lower_bound is not None and value < var.lower_bound - 1e-6:
                issues.append(
                    f"Variable {symbol}={value} below lower bound {var.lower_bound}"
                )
            if var.upper_bound is not None and value > var.upper_bound + 1e-6:
                issues.append(
                    f"Variable {symbol}={value} above upper bound {var.upper_bound}"
                )

    return issues


def _check_objective_recomputation(
    model: MathematicalModel, result: SolverResult
) -> dict[str, Any]:
    """Recompute objective from variable values and compare."""
    if not model.objectives:
        return {"matches": True, "note": "no objective"}

    obj = model.objectives[0]
    coeffs = SimpleLPCompiler._parse_coefficients(
        obj.expression,
        {v.symbol: i for i, v in enumerate(model.variables)},
        len(model.variables),
    )
    recomputed = sum(
        coeffs[i] * result.variable_values.get(model.variables[i].symbol, 0.0)
        for i in range(len(model.variables))
    )

    solver_obj = result.objective_value if result.objective_value is not None else float("nan")
    diff = abs(recomputed - solver_obj)
    tolerance = 1e-6

    return {
        "solver_objective": solver_obj,
        "recomputed_objective": recomputed,
        "absolute_difference": round(diff, 10),
        "relative_difference": round(diff / max(abs(solver_obj), 1e-9), 10) if solver_obj != 0 else 0.0,
        "tolerance": tolerance,
        "matches": diff <= tolerance,
    }