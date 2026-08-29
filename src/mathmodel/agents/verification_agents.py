"""MathModel AI — Sensitivity, Robustness, Red Team, and Model Repair agents."""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Optional

from mathmodel.domain.math_model import (
    MathematicalModel,
    Variable,
    Parameter,
    Objective,
    Constraint,
    ConstraintRelation,
    ObjectiveSense,
    VariableType,
    ParameterStatus,
)
from mathmodel.domain.verification import (
    SensitivityExperiment,
    SensitivityReport,
    SensitivityRun,
    RobustnessExperiment,
    RobustnessMethod,
    RobustnessReport,
    RedTeamIssue,
    RedTeamReport,
    RepairPlan,
    RepairType,
    ModelRevision,
    IssueSeverity,
    GateStatus,
    ValidationReport,
)
from mathmodel.solver import (
    SimpleLPCompiler,
    solve_lp_scipy,
    evaluate_constraints,
    SolverResult,
    SolverStatus,
)
from mathmodel.verification import build_validation_report, MathematicalValidationGate

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Sensitivity Agent (deterministic core)
# ═══════════════════════════════════════════════════════════════

class SensitivityAgent:
    """Runs sensitivity experiments. LLM designs, Python executes."""

    DEFAULT_PERTURBATIONS = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]

    def __init__(self, max_experiments: int = 5):
        self._max_experiments = max_experiments

    def run(
        self,
        model: MathematicalModel,
        parameter_ids: Optional[list[str]] = None,
    ) -> SensitivityReport:
        """Run sensitivity analysis on specified parameters."""
        # Select parameters: prefer estimated/uncertain ones
        if parameter_ids is None:
            target_params = self._select_parameters(model)
        else:
            target_params = [p for p in model.parameters if p.parameter_id in parameter_ids]

        target_params = target_params[:self._max_experiments]

        experiments = []
        for param in target_params:
            if param.value is None:
                continue
            experiment = self._run_experiment(model, param)
            experiments.append(experiment)

        # Summarize
        most_sensitive = []
        least_sensitive = []
        for exp in experiments:
            if exp.elasticity is not None:
                if abs(exp.elasticity) > 0.5:
                    most_sensitive.append(exp.parameter_id)
                else:
                    least_sensitive.append(exp.parameter_id)
            elif exp.status == GateStatus.WARNING:
                # Ineffective experiments must not be called "stable"
                # — they are unmeasurable; exclude from both lists

                pass

        return SensitivityReport(
            model_id=model.model_id,
            model_version=model.version,
            experiments=experiments,
            most_sensitive_parameters=most_sensitive,
            least_sensitive_parameters=least_sensitive,
            summary=f"{len(experiments)} experiments, {len(most_sensitive)} sensitive parameters",
        )

    def _select_parameters(self, model: MathematicalModel) -> list[Parameter]:
        """Select parameters for sensitivity: estimated first, then all."""
        estimated = [p for p in model.parameters if p.status == ParameterStatus.ESTIMATED]
        if estimated:
            return estimated
        return [p for p in model.parameters if p.value is not None]

    def _run_experiment(
        self, model: MathematicalModel, param: Parameter
    ) -> SensitivityExperiment:
        """Run perturbations for one parameter."""
        experiment = SensitivityExperiment(
            parameter_id=param.parameter_id,
            baseline=param.value,
        )

        baseline_value = param.value

        for delta in self.DEFAULT_PERTURBATIONS:
            perturbed = baseline_value * (1 + delta)
            # Clamp: don't allow negative values for non-negative parameters
            if baseline_value >= 0 and perturbed < 0:
                perturbed = 0.0

            run = self._solve_with_perturbation(model, param, perturbed, delta)
            experiment.runs.append(run)
            if delta == 0.0:
                experiment.perturbations.append(0.0)

        # Compute elasticity: % change in objective / % change in parameter
        baseline_run = next((r for r in experiment.runs if abs(r.delta_pct) < 1e-9), None)
        if baseline_run and baseline_run.objective_value and baseline_run.effective:
            # Use ±5% for elasticity (only from effective runs)
            plus5 = next((r for r in experiment.runs
                          if abs(r.delta_pct - 0.05) < 1e-9 and r.effective), None)
            if plus5 and plus5.objective_value:
                pct_change_obj = (plus5.objective_value - baseline_run.objective_value) / abs(baseline_run.objective_value)
                pct_change_param = 0.05
                experiment.elasticity = round(pct_change_obj / pct_change_param, 4)

        # Determine experiment status
        effective_runs = [r for r in experiment.runs if r.effective]
        if not effective_runs:
            # Perturbation had no effect on the compiled model — flag honestly
            experiment.status = GateStatus.WARNING
            experiment.interpretation = (
                "Perturbations did not change the compiled model. "
                "The parameter is likely not referenced by model expressions "
                "(numeric constants used instead of parameter symbols). "
                "Sensitivity result is NOT meaningful."
            )
        else:
            experiment.status = GateStatus.PASS if all(
                r.solver_status == "optimal" for r in effective_runs
            ) else GateStatus.WARNING

        return experiment

    def _solve_with_perturbation(
        self, model: MathematicalModel, param: Parameter, value: float, delta: float
    ) -> SensitivityRun:
        """Solve the model with a perturbed parameter value.

        Detects whether the perturbation actually changed the compiled model.
        """
        # Create a copy of the model with the parameter value changed
        modified = model.model_copy(deep=True)
        for p in modified.parameters:
            if p.parameter_id == param.parameter_id:
                p.value = value

        # Update expressions that reference the parameter symbol
        modified = self._apply_parameter_value(modified, param, value)

        # Compare compiled representation with baseline to verify effectiveness
        baseline_compiled = SimpleLPCompiler.compile(model)
        modified_compiled = SimpleLPCompiler.compile(modified)
        effective = self._compiled_differs(baseline_compiled, modified_compiled)

        modified_compiled["model_id"] = modified.model_id
        result = solve_lp_scipy(modified_compiled)

        return SensitivityRun(
            parameter_id=param.parameter_id,
            baseline=param.value,
            perturbed=value,
            delta_pct=round(delta, 4),
            objective_value=result.objective_value,
            variable_values=result.variable_values,
            solver_status=result.status.value,
            execution_real=result.execution_real,
            effective=effective,
        )

    @staticmethod
    def _compiled_differs(a: dict, b: dict) -> bool:
        """Compare two compiled LP representations."""
        for key in ("c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"):
            if a.get(key) != b.get(key):
                return True
        return False

    def _apply_parameter_value(
        self, model: MathematicalModel, param: Parameter, value: float
    ) -> MathematicalModel:
        """Substitute a parameter's value into model expressions.

        Handles both forms:
        1. Symbol form: expression references `p1` → replace `p1` with value
        2. Numeric form: expression embeds the baseline value as a constant
           (e.g. `30*x1` with p1 baseline 30) → replace the standalone
           constant token with the perturbed value
        """
        import re

        symbol = param.symbol
        baseline = param.value

        def substitute(expr: str) -> str:
            # 1. Symbol substitution
            if symbol in expr:
                expr = re.sub(
                    rf"\b{re.escape(symbol)}\b", str(value), expr
                )
            # 2. Numeric constant substitution (standalone baseline value)
            if baseline is not None:
                baseline_str = f"{baseline:g}"
                expr = re.sub(
                    rf"(?<![\w.]){re.escape(baseline_str)}(?![\w.])",
                    f"{value:g}",
                    expr,
                )
            return expr

        for obj in model.objectives:
            obj.expression = substitute(obj.expression)
        for con in model.constraints:
            con.expression = substitute(con.expression)
        return model

    def _apply_param(self, model, param, value):
        """Alias for _apply_parameter_value (used by robustness scenarios)."""
        return self._apply_parameter_value(model, param, value)


# ═══════════════════════════════════════════════════════════════
# Robustness Agent (deterministic core)
# ═══════════════════════════════════════════════════════════════

class RobustnessAgent:
    """Runs robustness experiments with reproducible seeds."""

    def __init__(self, default_seed: int = 42, default_samples: int = 100):
        self._default_seed = default_seed
        self._default_samples = default_samples

    def run(
        self,
        model: MathematicalModel,
        method: RobustnessMethod = RobustnessMethod.SCENARIO_ANALYSIS,
        uncertain_parameters: Optional[list[str]] = None,
    ) -> RobustnessReport:
        """Run robustness analysis."""
        experiments = []

        if method == RobustnessMethod.SCENARIO_ANALYSIS:
            experiments.append(self._run_scenarios(model, uncertain_parameters))
        elif method == RobustnessMethod.MONTE_CARLO:
            experiments.append(self._run_monte_carlo(model, uncertain_parameters))
        elif method == RobustnessMethod.NOISE_PERTURBATION:
            experiments.append(self._run_noise(model, uncertain_parameters))
        else:
            experiments.append(self._run_scenarios(model, uncertain_parameters))

        return RobustnessReport(
            model_id=model.model_id,
            model_version=model.version,
            experiments=experiments,
            summary=f"{len(experiments)} robustness experiments",
        )

    def _run_scenarios(
        self, model: MathematicalModel, uncertain_parameters: Optional[list[str]]
    ) -> RobustnessExperiment:
        """Run base/optimistic/pessimistic scenarios."""
        params = [p for p in model.parameters if p.value is not None]
        if uncertain_parameters:
            params = [p for p in params if p.parameter_id in uncertain_parameters]

        scenarios = [
            {"name": "base", "delta": 0.0},
            {"name": "optimistic", "delta": 0.20},
            {"name": "pessimistic", "delta": -0.20},
            {"name": "stress", "delta": -0.50},
        ]

        baseline_compiled = SimpleLPCompiler.compile(model)
        results = []
        failures = 0
        effective_count = 0
        for scenario in scenarios:
            modified = model.model_copy(deep=True)
            for p in modified.parameters:
                base_val = next(
                    (pp.value for pp in params if pp.parameter_id == p.parameter_id),
                    None,
                )
                if base_val is not None:
                    new_val = base_val * (1 + scenario["delta"])
                    if base_val >= 0 and new_val < 0:
                        new_val = 0.0
                    p.value = new_val
                    modified = self._apply_param(modified, p, new_val)

            compiled = SimpleLPCompiler.compile(modified)
            effective = any(
                compiled.get(k) != baseline_compiled.get(k)
                for k in ("c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds")
            )
            if effective:
                effective_count += 1

            compiled["model_id"] = modified.model_id
            result = solve_lp_scipy(compiled)
            if result.status != SolverStatus.OPTIMAL:
                failures += 1
            results.append({
                "scenario": scenario["name"],
                "objective": result.objective_value,
                "status": result.status.value,
                "effective": effective,
            })

        if effective_count == 0:
            summary = (
                "Perturbations had no effect on the compiled model — "
                "expressions likely use numeric constants instead of parameter "
                "symbols. Robustness result is NOT meaningful."
            )
        else:
            summary = f"{failures} failures in {len(scenarios)} scenarios"

        return RobustnessExperiment(
            method=RobustnessMethod.SCENARIO_ANALYSIS,
            uncertain_inputs=[p.parameter_id for p in params],
            scenarios=scenarios,
            sample_count=len(scenarios),
            seed=self._default_seed,
            results=results,
            failure_rate=failures / len(scenarios) if scenarios else 0.0,
            metrics={"effective_scenarios": effective_count},
            summary=summary,
        )

    def _run_monte_carlo(
        self, model: MathematicalModel, uncertain_parameters: Optional[list[str]]
    ) -> RobustnessExperiment:
        """Run Monte Carlo with reproducible seed."""
        rng = random.Random(self._default_seed)
        params = [p for p in model.parameters if p.value is not None]
        if uncertain_parameters:
            params = [p for p in params if p.parameter_id in uncertain_parameters]

        results = []
        failures = 0
        objectives = []
        for _ in range(self._default_samples):
            modified = model.model_copy(deep=True)
            for p in modified.parameters:
                base_val = next((pp.value for pp in params if pp.parameter_id == p.parameter_id), None)
                if base_val is not None:
                    # Normal noise: ±10%
                    new_val = base_val * (1 + rng.gauss(0, 0.10))
                    if base_val >= 0 and new_val < 0:
                        new_val = 0.0
                    p.value = new_val
                    modified = self._apply_param(modified, p, new_val)

            compiled = SimpleLPCompiler.compile(modified)
            compiled["model_id"] = modified.model_id
            result = solve_lp_scipy(compiled)
            if result.status != SolverStatus.OPTIMAL:
                failures += 1
            else:
                objectives.append(result.objective_value)
            results.append({"objective": result.objective_value, "status": result.status.value})

        metrics = {}
        if objectives:
            metrics["mean_objective"] = sum(objectives) / len(objectives)
            metrics["min_objective"] = min(objectives)
            metrics["max_objective"] = max(objectives)
            metrics["std_objective"] = _std(objectives)

        return RobustnessExperiment(
            method=RobustnessMethod.MONTE_CARLO,
            uncertain_inputs=[p.parameter_id for p in params],
            sample_count=self._default_samples,
            seed=self._default_seed,
            results=results,
            failure_rate=failures / self._default_samples,
            metrics=metrics,
            summary=f"MC: {failures} failures in {self._default_samples} samples",
        )

    def _run_noise(
        self, model: MathematicalModel, uncertain_parameters: Optional[list[str]]
    ) -> RobustnessExperiment:
        """Run noise perturbation (same as MC but fixed seed)."""
        return self._run_monte_carlo(model, uncertain_parameters)

    def _apply_param(self, model, param, value):
        """Substitute parameter value (symbol or numeric constant form)."""
        import re
        symbol = param.symbol
        baseline = param.value

        def substitute(expr: str) -> str:
            if symbol in expr:
                expr = re.sub(rf"\b{re.escape(symbol)}\b", str(value), expr)
            if baseline is not None:
                baseline_str = f"{baseline:g}"
                expr = re.sub(
                    rf"(?<![\w.]){re.escape(baseline_str)}(?![\w.])",
                    f"{value:g}",
                    expr,
                )
            return expr

        for obj in model.objectives:
            obj.expression = substitute(obj.expression)
        for con in model.constraints:
            con.expression = substitute(con.expression)
        return model


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


# ═══════════════════════════════════════════════════════════════
# Red Team Agent (deterministic detection)
# ═══════════════════════════════════════════════════════════════

class RedTeamAgent:
    """Adversarially reviews the model and results."""

    def review(
        self,
        model: MathematicalModel,
        result: SolverResult,
        validation: ValidationReport,
        sensitivity: Optional[SensitivityReport] = None,
        robustness: Optional[RobustnessReport] = None,
    ) -> RedTeamReport:
        """Review the model adversarially."""
        issues: list[RedTeamIssue] = []

        # 1. Check validation failures
        for v_issue in validation.issues:
            if v_issue.severity == IssueSeverity.CRITICAL:
                issues.append(RedTeamIssue(
                    severity=IssueSeverity.CRITICAL,
                    category="validation",
                    title=f"Validation failure: {v_issue.message}",
                    description=v_issue.message,
                    evidence=[{"validation_issue_id": v_issue.issue_id}],
                    suggested_fix="Fix the underlying issue and re-validate",
                    target_component="validation",
                    confidence=0.95,
                ))

        # 2. Check solver status
        if result.status != SolverStatus.OPTIMAL:
            issues.append(RedTeamIssue(
                severity=IssueSeverity.CRITICAL,
                category="solver",
                title=f"Solver status is {result.status.value}, not OPTIMAL",
                description="The model did not reach optimality",
                evidence=[{"solver_status": result.status.value}],
                suggested_fix="Check model feasibility or solver configuration",
                target_component="solver",
                confidence=0.9,
            ))

        # 3. Check constraint violations
        constraint_eval = validation.constraint_validation
        if constraint_eval and not constraint_eval.get("all_satisfied", True):
            issues.append(RedTeamIssue(
                severity=IssueSeverity.CRITICAL,
                category="constraints",
                title=f"Constraints violated: {constraint_eval.get('violated_constraints')}",
                description=f"max_violation={constraint_eval.get('max_violation')}",
                evidence=[{"constraint_eval": constraint_eval}],
                suggested_fix="Add or fix constraints",
                target_component="constraints",
                confidence=0.95,
            ))

        # 4. Check sensitivity instability
        if sensitivity and sensitivity.most_sensitive_parameters:
            issues.append(RedTeamIssue(
                severity=IssueSeverity.MAJOR,
                category="sensitivity",
                title=f"Highly sensitive parameters: {sensitivity.most_sensitive_parameters}",
                description="Small parameter changes cause large result changes",
                evidence=[{"parameters": sensitivity.most_sensitive_parameters}],
                suggested_fix="Document uncertainty or add robustness measures",
                target_component="parameters",
                confidence=0.7,
            ))

        # 5. Check robustness failure rate
        if robustness:
            for exp in robustness.experiments:
                if exp.failure_rate > 0.2:
                    issues.append(RedTeamIssue(
                        severity=IssueSeverity.MAJOR,
                        category="robustness",
                        title=f"High failure rate ({exp.failure_rate:.0%}) in {exp.method.value}",
                        description="Model is unstable under uncertainty",
                        evidence=[{"failure_rate": exp.failure_rate}],
                        suggested_fix="Add robustness constraints or improve parameter estimates",
                        target_component="robustness",
                        confidence=0.8,
                    ))

        # 6. Check execution truth
        if not result.execution_real:
            issues.append(RedTeamIssue(
                severity=IssueSeverity.CRITICAL,
                category="execution",
                title="Result is not from real execution",
                description="SolverResult has execution_real=False",
                evidence=[],
                suggested_fix="Run real execution",
                target_component="execution",
                confidence=1.0,
            ))

        return RedTeamReport(
            model_id=model.model_id,
            issues=issues,
            summary=f"{len(issues)} issues found ({sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)} critical)",
        )


# ═══════════════════════════════════════════════════════════════
# Model Repair Agent
# ═══════════════════════════════════════════════════════════════

class ModelRepairAgent:
    """Repairs models based on validation and red team findings."""

    MAX_REPAIR_ITERATIONS = 3

    def __init__(self):
        self._repair_count = 0

    def create_repair_plan(
        self,
        model: MathematicalModel,
        redteam: RedTeamReport,
        validation: ValidationReport,
    ) -> RepairPlan:
        """Create a repair plan from findings."""
        self._repair_count += 1

        issue_ids = [i.issue_id for i in redteam.issues]
        root_causes = [i.title for i in redteam.issues]

        # Determine repair type
        repair_type = RepairType.CONSTRAINT_FIX
        affected_constraints = []

        for issue in redteam.issues:
            if issue.category == "constraints":
                repair_type = RepairType.CONSTRAINT_FIX
                if issue.evidence:
                    for ev in issue.evidence:
                        if "constraint_eval" in ev:
                            affected_constraints.extend(
                                ev["constraint_eval"].get("violated_constraints", [])
                            )
            elif issue.category == "solver" and issue.evidence:
                for ev in issue.evidence:
                    if ev.get("solver_status") == "infeasible":
                        repair_type = RepairType.MODEL_STRUCTURE_FIX

        return RepairPlan(
            issue_ids=issue_ids,
            root_causes=root_causes,
            proposed_changes=[f"Fix: {rc}" for rc in root_causes],
            repair_type=repair_type,
            affected_constraints=affected_constraints,
            expected_effect="Constraints satisfied after repair",
            risk="medium" if repair_type == RepairType.MODEL_STRUCTURE_FIX else "low",
            requires_model_switch=(repair_type == RepairType.MODEL_STRUCTURE_FIX),
        )

    def apply_repair(
        self,
        model: MathematicalModel,
        plan: RepairPlan,
    ) -> MathematicalModel:
        """Apply the repair plan to the model.

        Only applies structured RepairActions. Every action that introduces
        a new numeric value MUST carry a source_reference. Actions without
        provenance are rejected.
        """
        if plan.requires_model_switch:
            # Don't silently switch — signal the caller
            raise ModelSwitchRequired("Model switch required — defer to ModelJury or human")

        modified = model.model_copy(deep=True)

        if not plan.actions:
            # No structured actions → refuse to guess
            raise RepairSourceRequired(
                f"RepairPlan {plan.plan_id} has no structured actions. "
                "A repair must specify concrete, source-backed changes."
            )

        for action in plan.actions:
            if action.kind == "parameter_set":
                if action.parameter_id is None or action.new_value is None:
                    raise RepairSourceRequired("parameter_set action missing parameter_id/new_value")
                if not action.source_reference:
                    raise RepairSourceRequired(
                        f"Action {action.action_id} sets a new value without "
                        "source_reference. Repairs must cite evidence."
                    )
                found = False
                old_value = None
                for p in modified.parameters:
                    if p.parameter_id == action.parameter_id:
                        old_value = p.value  # Capture old value BEFORE overwriting
                        p.value = action.new_value
                        p.source_reference = action.source_reference
                        found = True
                if not found:
                    raise RepairSourceRequired(
                        f"Action {action.action_id} references unknown parameter "
                        f"{action.parameter_id}"
                    )
                modified = self._substitute_parameter(
                    modified, action.parameter_id, action.new_value, old_value
                )

            elif action.kind == "constraint_add":
                if not action.constraint_expression or action.constraint_rhs is None:
                    raise RepairSourceRequired("constraint_add action missing expression/rhs")
                if not action.source_reference:
                    raise RepairSourceRequired(
                        f"Action {action.action_id} adds a constraint without "
                        "source_reference. Repairs must cite evidence."
                    )
                from mathmodel.domain.math_model import Constraint, ConstraintRelation
                relation = {
                    "le": ConstraintRelation.LE,
                    "ge": ConstraintRelation.GE,
                    "eq": ConstraintRelation.EQ,
                }.get((action.constraint_relation or "le").lower(), ConstraintRelation.LE)
                new_constraint = Constraint(
                    constraint_id=action.constraint_id or f"CON-REPAIR-{len(modified.constraints) + 1}",
                    name=f"Repair constraint {len(modified.constraints) + 1}",
                    expression=action.constraint_expression,
                    relation=relation,
                    rhs=action.constraint_rhs,
                    source=action.source_reference,
                )
                modified.constraints.append(new_constraint)

            else:
                raise RepairSourceRequired(f"Unsupported repair action kind: {action.kind}")

        modified.version += 1
        modified.metadata = dict(modified.metadata or {})
        modified.metadata["repair_plan_id"] = plan.plan_id
        modified.metadata["repair_iterations"] = self._repair_count

        return modified

    def _substitute_parameter(
        self, model: MathematicalModel, parameter_id: str, value: float,
        old_value: Optional[float] = None,
    ) -> MathematicalModel:
        """Substitute a parameter's new value into model expressions."""
        import re

        for p in model.parameters:
            if p.parameter_id == parameter_id:
                symbol = p.symbol
                # Substitute symbol references
                for obj in model.objectives:
                    if symbol in obj.expression:
                        obj.expression = re.sub(
                            rf"\b{re.escape(symbol)}\b", str(value), obj.expression
                        )
                for con in model.constraints:
                    if symbol in con.expression:
                        con.expression = re.sub(
                            rf"\b{re.escape(symbol)}\b", str(value), con.expression
                        )
                # Also substitute the old baseline numeric constant if present
                if old_value is not None:
                    old_str = f"{old_value:g}"
                    for obj in model.objectives:
                        obj.expression = re.sub(
                            rf"(?<![\w.]){re.escape(old_str)}(?![\w.])",
                            f"{value:g}",
                            obj.expression,
                        )
                    for con in model.constraints:
                        con.expression = re.sub(
                            rf"(?<![\w.]){re.escape(old_str)}(?![\w.])",
                            f"{value:g}",
                            con.expression,
                        )
        return model

    @property
    def repair_count(self) -> int:
        return self._repair_count

    @property
    def max_reached(self) -> bool:
        return self._repair_count >= self.MAX_REPAIR_ITERATIONS


class RepairSourceRequired(Exception):
    """Raised when a repair action lacks source provenance or is invalid."""
    pass


class ModelSwitchRequired(Exception):
    """Raised when a repair requires switching to a different model."""
    pass


# ═══════════════════════════════════════════════════════════════
# Verification Quality Gate
# ═══════════════════════════════════════════════════════════════

class VerificationQualityGate:
    """Deterministic quality gate over verification results."""

    @staticmethod
    def evaluate(
        validation: ValidationReport,
        redteam: RedTeamReport,
        sensitivity: Optional[SensitivityReport] = None,
        robustness: Optional[RobustnessReport] = None,
    ) -> str:
        """Return VerificationStatus value."""
        from mathmodel.domain.verification import VerificationStatus

        # Hard failures
        if validation.overall_status == GateStatus.FAIL:
            return VerificationStatus.FAILED.value

        if redteam.critical_count > 0:
            return VerificationStatus.REPAIR_REQUIRED.value

        # Robustness failure
        if robustness:
            for exp in robustness.experiments:
                if exp.failure_rate > 0.5:
                    return VerificationStatus.FAILED.value

        # Warnings
        if validation.overall_status == GateStatus.WARNING:
            return VerificationStatus.VERIFIED_WITH_WARNINGS.value

        if redteam.issues:
            return VerificationStatus.VERIFIED_WITH_WARNINGS.value

        return VerificationStatus.VERIFIED.value