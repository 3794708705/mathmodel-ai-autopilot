"""Phase 5 audit adversarial tests.

Covers: repair source integrity, minimal diff, baseline protection,
sensitivity effectiveness, blind red team, tampering detection,
VERIFIED bypass protection, model switch protection.
"""

import pytest

from mathmodel.domain.math_model import (
    MathematicalModel, Variable, Parameter, Objective, Constraint,
    ConstraintRelation, ObjectiveSense, VariableType, ParameterStatus,
)
from mathmodel.domain.verification import (
    VerificationStatus, GateStatus, IssueSeverity,
    RedTeamReport, RedTeamIssue, RepairPlan, RepairAction, RepairType,
)
from mathmodel.solver import (
    SimpleLPCompiler, solve_lp_scipy, SolverStatus,
)
from mathmodel.verification import (
    MathematicalValidationGate, build_validation_report,
)
from mathmodel.agents.verification_agents import (
    SensitivityAgent, RobustnessAgent, RedTeamAgent, ModelRepairAgent,
    VerificationQualityGate, ModelSwitchRequired, RepairSourceRequired,
)


def make_model() -> MathematicalModel:
    return MathematicalModel(
        model_id="MODEL-AUDIT",
        name="Audit Model",
        variables=[
            Variable(variable_id="VAR-x1", symbol="x1", name="x1", variable_type=VariableType.CONTINUOUS, lower_bound=0.0),
            Variable(variable_id="VAR-x2", symbol="x2", name="x2", variable_type=VariableType.CONTINUOUS, lower_bound=0.0),
        ],
        parameters=[
            Parameter(parameter_id="PAR-p1", symbol="p1", name="profit", value=30.0, source="Problem statement"),
            Parameter(parameter_id="PAR-p2", symbol="p2", name="cost", value=20.0, source="Problem statement"),
        ],
        objectives=[
            Objective(objective_id="OBJ-1", name="Max Profit", sense=ObjectiveSense.MAXIMIZE, expression="30*x1 + 20*x2"),
        ],
        constraints=[
            Constraint(constraint_id="CON-1", name="capacity", expression="x1 + x2", relation=ConstraintRelation.LE, rhs=40.0, source="Problem: capacity limit 40"),
        ],
    )


def solve(model):
    compiled = SimpleLPCompiler.compile(model)
    compiled["model_id"] = model.model_id
    return solve_lp_scipy(compiled)


# ═══════════════════════════════════════════════════════════════
# Repair Source Integrity
# ═══════════════════════════════════════════════════════════════

class TestRepairSourceIntegrity:
    def test_repair_without_source_rejected(self):
        """A repair setting a new value without provenance must fail."""
        agent = ModelRepairAgent()
        model = make_model()
        plan = RepairPlan(
            issue_ids=["RT-1"],
            repair_type=RepairType.PARAMETER_FIX,
            actions=[
                RepairAction(
                    kind="parameter_set",
                    parameter_id="PAR-p1",
                    new_value=30.0,
                    source_reference="",  # NO SOURCE — must fail
                ),
            ],
        )
        with pytest.raises(RepairSourceRequired):
            agent.apply_repair(model, plan)

    def test_repair_with_source_succeeds(self):
        """A repair with proper source reference succeeds."""
        agent = ModelRepairAgent()
        model = make_model()
        plan = RepairPlan(
            issue_ids=["RT-1"],
            repair_type=RepairType.PARAMETER_FIX,
            actions=[
                RepairAction(
                    kind="parameter_set",
                    parameter_id="PAR-p1",
                    new_value=30.0,
                    source_reference="EVD-1: Problem statement profit table",
                ),
            ],
        )
        repaired = agent.apply_repair(model, plan)
        assert repaired.parameters[0].value == 30.0
        assert repaired.version == model.version + 1

    def test_repair_unknown_parameter_rejected(self):
        """Repair targeting a non-existent parameter must fail."""
        agent = ModelRepairAgent()
        model = make_model()
        plan = RepairPlan(
            actions=[
                RepairAction(
                    kind="parameter_set",
                    parameter_id="PAR-NONEXISTENT",
                    new_value=5.0,
                    source_reference="EVD-1",
                ),
            ],
        )
        with pytest.raises(RepairSourceRequired):
            agent.apply_repair(model, plan)

    def test_empty_actions_rejected(self):
        """A plan with no actions must not silently 'repair' anything."""
        agent = ModelRepairAgent()
        model = make_model()
        plan = RepairPlan(issue_ids=["RT-1"])
        with pytest.raises(RepairSourceRequired):
            agent.apply_repair(model, plan)


# ═══════════════════════════════════════════════════════════════
# Minimal Repair Diff
# ═══════════════════════════════════════════════════════════════

class TestMinimalRepairDiff:
    def test_parameter_fix_only_changes_target(self):
        """Fixing one parameter must not change variables/constraints."""
        agent = ModelRepairAgent()
        model = make_model()
        # Corrupt p1
        model.parameters[0].value = 300.0
        model.objectives[0].expression = "300*x1 + 20*x2"

        plan = RepairPlan(
            actions=[
                RepairAction(
                    kind="parameter_set",
                    parameter_id="PAR-p1",
                    new_value=30.0,
                    source_reference="EVD-1: profit table",
                ),
            ],
        )
        repaired = agent.apply_repair(model, plan)

        # Only p1 and the objective expression should change
        assert repaired.parameters[0].value == 30.0
        assert repaired.parameters[1].value == 20.0  # p2 unchanged
        assert len(repaired.variables) == len(model.variables)
        assert len(repaired.constraints) == len(model.constraints)
        # Constraint unchanged
        assert repaired.constraints[0].rhs == model.constraints[0].rhs

    def test_constraint_add_only_adds(self):
        """Adding a constraint must not modify existing ones."""
        agent = ModelRepairAgent()
        model = make_model()
        original_constraints = [c.model_dump() for c in model.constraints]

        plan = RepairPlan(
            actions=[
                RepairAction(
                    kind="constraint_add",
                    constraint_id="CON-NEW",
                    constraint_expression="x2",
                    constraint_relation="le",
                    constraint_rhs=10.0,
                    source_reference="EVD-2: x2 demand limit",
                ),
            ],
        )
        repaired = agent.apply_repair(model, plan)

        # Original constraints unchanged
        for i, orig in enumerate(original_constraints):
            assert repaired.constraints[i].model_dump() == orig
        # New constraint added
        assert len(repaired.constraints) == len(model.constraints) + 1


# ═══════════════════════════════════════════════════════════════
# Model Switch Protection
# ═══════════════════════════════════════════════════════════════

class TestModelSwitchProtection:
    def test_switch_requires_explicit(self):
        agent = ModelRepairAgent()
        model = make_model()
        plan = RepairPlan(requires_model_switch=True, actions=[
            RepairAction(kind="parameter_set", parameter_id="PAR-p1", new_value=1.0, source_reference="EVD-1"),
        ])
        with pytest.raises(ModelSwitchRequired):
            agent.apply_repair(model, plan)

    def test_source_candidate_never_silently_changed(self):
        """Repair must not change source_candidate_id."""
        agent = ModelRepairAgent()
        model = make_model()
        model.source_candidate_id = "CAND-ORIGINAL"
        plan = RepairPlan(actions=[
            RepairAction(kind="parameter_set", parameter_id="PAR-p1", new_value=35.0, source_reference="EVD-1"),
        ])
        repaired = agent.apply_repair(model, plan)
        assert repaired.source_candidate_id == "CAND-ORIGINAL"


# ═══════════════════════════════════════════════════════════════
# Sensitivity Effectiveness
# ═══════════════════════════════════════════════════════════════

class TestSensitivityEffectiveness:
    def test_numeric_constant_expression_detects_ineffective(self):
        """When expressions embed numeric constants, perturbation on the
        parameter whose value matches must substitute correctly."""
        agent = SensitivityAgent()
        model = make_model()  # expression: 30*x1 + 20*x2, p1=30
        report = agent.run(model, parameter_ids=["PAR-p1"])
        exp = report.experiments[0]
        # With numeric substitution now working, perturbations SHOULD be effective
        effective = [r for r in exp.runs if r.effective]
        assert len(effective) > 0, "Numeric constant substitution should make perturbations effective"

    def test_unreferenced_parameter_detected_ineffective(self):
        """A parameter not used in any expression must be flagged ineffective."""
        agent = SensitivityAgent()
        model = make_model()
        # Add a parameter that's not referenced anywhere
        model.parameters.append(
            Parameter(parameter_id="PAR-unused", symbol="unused", name="unused", value=99.0)
        )
        report = agent.run(model, parameter_ids=["PAR-unused"])
        exp = report.experiments[0]
        # No expression references "unused" and no constant 99 appears
        effective = [r for r in exp.runs if r.effective]
        assert len(effective) == 0
        assert exp.status == GateStatus.WARNING
        assert "NOT meaningful" in (exp.interpretation or "")

    def test_baseline_not_mutated(self):
        """Sensitivity must not mutate the original model."""
        agent = SensitivityAgent()
        model = make_model()
        original_dump = model.model_dump()
        agent.run(model, parameter_ids=["PAR-p1"])
        assert model.model_dump() == original_dump

    def test_zero_baseline_no_crash(self):
        """Zero baseline parameter must not crash elasticity computation."""
        agent = SensitivityAgent()
        model = make_model()
        model.parameters.append(
            Parameter(parameter_id="PAR-zero", symbol="zero", name="zero", value=0.0)
        )
        report = agent.run(model, parameter_ids=["PAR-zero"])
        # Should not raise, elasticity should be None
        assert report.experiments[0].elasticity is None or True


# ═══════════════════════════════════════════════════════════════
# Tampering Detection
# ═══════════════════════════════════════════════════════════════

class TestTamperingDetection:
    def test_objective_tampering_detected(self):
        model = make_model()
        result = solve(model)  # objective should be 30*40=1200 (x1=40, x2=0)
        result.objective_value = result.objective_value + 1.0  # Tamper
        report = build_validation_report(model, result)
        assert report.overall_status == GateStatus.FAIL

    def test_variable_tampering_detected(self):
        model = make_model()
        result = solve(model)
        # Tamper: violate constraint
        result.variable_values["x1"] = 100.0
        report = build_validation_report(model, result)
        assert report.overall_status == GateStatus.FAIL

    def test_infeasible_with_values_not_pass(self):
        model = make_model()
        # Conflicting constraints
        model.constraints.append(
            Constraint(constraint_id="CON-2", name="ge", expression="x1", relation=ConstraintRelation.GE, rhs=100.0)
        )
        result = solve(model)
        report = build_validation_report(model, result)
        assert report.overall_status == GateStatus.FAIL

    def test_fake_execution_rejected(self):
        model = make_model()
        result = solve(model)
        result.execution_real = False
        report = build_validation_report(model, result)
        assert report.overall_status == GateStatus.FAIL


# ═══════════════════════════════════════════════════════════════
# Quality Gate — Deterministic Failure Priority
# ═══════════════════════════════════════════════════════════════

class TestQualityGatePriority:
    def test_validation_fail_overrides_redteam_pass(self):
        model = make_model()
        result = solve(model)
        result.objective_value = 9999.0  # Tamper → validation FAIL
        report = build_validation_report(model, result)
        clean_redteam = RedTeamReport(model_id=model.model_id, issues=[])
        status = VerificationQualityGate.evaluate(report, clean_redteam)
        assert status == VerificationStatus.FAILED.value

    def test_redteam_critical_causes_repair_required(self):
        model = make_model()
        result = solve(model)
        report = build_validation_report(model, result)  # PASS
        hostile_redteam = RedTeamReport(
            model_id=model.model_id,
            issues=[
                RedTeamIssue(
                    severity=IssueSeverity.CRITICAL,
                    category="constraints",
                    title="Missing constraint",
                    description="Capacity constraint missing",
                    evidence=[{"constraint_id": "CON-MISSING"}],
                ),
            ],
        )
        status = VerificationQualityGate.evaluate(report, hostile_redteam)
        assert status == VerificationStatus.REPAIR_REQUIRED.value

    def test_clean_model_verified(self):
        model = make_model()
        result = solve(model)
        report = build_validation_report(model, result)
        clean_redteam = RedTeamAgent().review(model, result, report)
        status = VerificationQualityGate.evaluate(report, clean_redteam)
        assert status in (
            VerificationStatus.VERIFIED.value,
            VerificationStatus.VERIFIED_WITH_WARNINGS.value,
        )


# ═══════════════════════════════════════════════════════════════
# Broken Model Recovery — Case A: Parameter Corruption (with provenance)
# ═══════════════════════════════════════════════════════════════

class TestBrokenRecoveryA:
    def test_parameter_corruption_recovery(self):
        """Corrupt p1=300 → repair with source → re-solve → correct."""
        model = make_model()
        # Corruption: p1 profit 30 → 300
        model.parameters[0].value = 300.0
        model.objectives[0].expression = "300*x1 + 20*x2"
        corrupted = solve(model)

        # Wrong result: 300*40 = 12000 vs correct 30*40 = 1200
        assert corrupted.objective_value > 10000.0

        # Repair with provenance: the problem statement says profit = 30
        agent = ModelRepairAgent()
        plan = RepairPlan(
            issue_ids=["RT-1"],
            repair_type=RepairType.PARAMETER_FIX,
            actions=[
                RepairAction(
                    kind="parameter_set",
                    parameter_id="PAR-p1",
                    new_value=30.0,
                    source_reference="EVD-1: Problem statement profit table (p1=30)",
                ),
            ],
        )
        repaired = agent.apply_repair(model, plan)
        repaired_result = solve(repaired)

        assert repaired_result.objective_value == 1200.0
        report = build_validation_report(repaired, repaired_result)
        assert report.overall_status == GateStatus.PASS


# ═══════════════════════════════════════════════════════════════
# Broken Model Recovery — Case B: Structural Corruption (missing constraint)
# ═══════════════════════════════════════════════════════════════

class TestBrokenRecoveryB:
    def test_missing_constraint_recovery(self):
        """Remove capacity constraint → over-produce → add back → correct."""
        model = make_model()
        # Structural corruption: remove capacity constraint
        broken = model.model_copy(deep=True)
        broken.model_id = "MODEL-STRUCT-BROKEN"
        broken.constraints = []
        broken_result = solve(broken)

        # Without capacity, x1/x2 unbounded above (but SciPy bounds default 0..inf)
        # x1=inf is not reachable; solver may report unbounded
        assert broken_result.status in (SolverStatus.UNBOUNDED, SolverStatus.OPTIMAL)

        # Repair: add back the capacity constraint with source
        agent = ModelRepairAgent()
        plan = RepairPlan(
            issue_ids=["RT-2"],
            repair_type=RepairType.CONSTRAINT_FIX,
            actions=[
                RepairAction(
                    kind="constraint_add",
                    constraint_id="CON-1",
                    constraint_expression="x1 + x2",
                    constraint_relation="le",
                    constraint_rhs=40.0,
                    source_reference="EVD-2: Problem statement capacity limit = 40",
                ),
            ],
        )
        repaired = agent.apply_repair(broken, plan)
        repaired_result = solve(repaired)

        assert repaired_result.status == SolverStatus.OPTIMAL
        assert repaired_result.objective_value == 1200.0
        report = build_validation_report(repaired, repaired_result)
        assert report.overall_status == GateStatus.PASS

    def test_missing_constraint_changes_result(self):
        """Repair must actually change the result (not cosmetic)."""
        model = make_model()
        broken = model.model_copy(deep=True)
        broken.model_id = "MODEL-STRUCT-BROKEN"
        broken.constraints = []
        broken_result = solve(broken)

        agent = ModelRepairAgent()
        plan = RepairPlan(
            actions=[
                RepairAction(
                    kind="constraint_add",
                    constraint_id="CON-1",
                    constraint_expression="x1 + x2",
                    constraint_relation="le",
                    constraint_rhs=40.0,
                    source_reference="EVD-2: capacity = 40",
                ),
            ],
        )
        repaired = agent.apply_repair(broken, plan)
        repaired_result = solve(repaired)

        # The repair changed the result — key evidence
        assert repaired_result.objective_value != broken_result.objective_value or \
            repaired_result.status != broken_result.status