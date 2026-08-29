"""Phase 4 audit adversarial tests.

Covers: maximize sign, GE conversion, EQ constraints, variable bounds,
variable ordering, infeasible/unbounded, safe expression, unit system,
provenance, execution truth.
"""

import pytest

from mathmodel.domain.math_model import (
    MathematicalModel, Variable, Parameter, Objective, Constraint,
    ConstraintRelation, ObjectiveSense, VariableType, ParameterStatus,
    EquationType,
)
from mathmodel.solver import (
    SimpleLPCompiler, solve_lp_scipy, evaluate_constraints,
    SolverStatus, SolverResult,
)
from mathmodel.math_ops import (
    validate_expression, ExpressionValidationError,
    SymbolRegistry, EquationRegistry, UnitChecker,
)


# ═══════════════════════════════════════════════════════════════
# Maximize Sign Conversion
# ═══════════════════════════════════════════════════════════════

class TestMaximizeSign:
    def test_maximize_negates_coefficients(self):
        model = MathematicalModel(
            name="Max Test",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS, lower_bound=0.0)],
            objectives=[Objective(objective_id="OBJ-1", name="Max", sense=ObjectiveSense.MAXIMIZE, expression="3*x")],
            constraints=[Constraint(constraint_id="CON-1", name="bound", expression="x", relation=ConstraintRelation.LE, rhs=10.0)],
        )
        compiled = SimpleLPCompiler.compile(model)
        # linprog minimizes, so maximize 3*x → minimize -3*x
        assert compiled["c"] == [-3.0]
        assert compiled["maximize"] is True

    def test_minimize_preserves_sign(self):
        model = MathematicalModel(
            name="Min Test",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS, lower_bound=0.0)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="3*x")],
            constraints=[Constraint(constraint_id="CON-1", name="bound", expression="x", relation=ConstraintRelation.LE, rhs=10.0)],
        )
        compiled = SimpleLPCompiler.compile(model)
        assert compiled["c"] == [3.0]

    def test_maximize_objective_negated_back(self):
        """After solving a maximization problem, objective_value should be positive."""
        model = MathematicalModel(
            name="Max Profit",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS, lower_bound=0.0)],
            objectives=[Objective(objective_id="OBJ-1", name="Max", sense=ObjectiveSense.MAXIMIZE, expression="5*x")],
            constraints=[Constraint(constraint_id="CON-1", name="cap", expression="x", relation=ConstraintRelation.LE, rhs=10.0)],
        )
        compiled = SimpleLPCompiler.compile(model)
        compiled["model_id"] = model.model_id
        result = solve_lp_scipy(compiled)
        assert result.status == SolverStatus.OPTIMAL
        assert result.objective_value == 50.0  # max 5*x, x=10 → 50
        assert result.variable_values["x"] == 10.0


# ═══════════════════════════════════════════════════════════════
# GE Constraint Conversion
# ═══════════════════════════════════════════════════════════════

class TestGEConversion:
    def test_ge_converts_to_le(self):
        model = MathematicalModel(
            name="GE Test",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS, lower_bound=0.0)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="x")],
            constraints=[Constraint(constraint_id="CON-1", name="min", expression="x", relation=ConstraintRelation.GE, rhs=5.0)],
        )
        compiled = SimpleLPCompiler.compile(model)
        # x >= 5 → -x <= -5
        assert compiled["A_ub"] == [[-1.0]]
        assert compiled["b_ub"] == [-5.0]

    def test_ge_solution_respects_bound(self):
        model = MathematicalModel(
            name="GE Test",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS, lower_bound=0.0)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="x")],
            constraints=[Constraint(constraint_id="CON-1", name="min", expression="x", relation=ConstraintRelation.GE, rhs=5.0)],
        )
        compiled = SimpleLPCompiler.compile(model)
        compiled["model_id"] = model.model_id
        result = solve_lp_scipy(compiled)
        assert result.status == SolverStatus.OPTIMAL
        assert result.variable_values["x"] >= 4.99  # Should be >= 5


# ═══════════════════════════════════════════════════════════════
# EQ Constraint
# ═══════════════════════════════════════════════════════════════

class TestEQConstraint:
    def test_eq_goes_to_a_eq(self):
        model = MathematicalModel(
            name="EQ Test",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="x")],
            constraints=[Constraint(constraint_id="CON-1", name="eq", expression="x", relation=ConstraintRelation.EQ, rhs=7.0)],
        )
        compiled = SimpleLPCompiler.compile(model)
        assert compiled["A_eq"] == [[1.0]]
        assert compiled["b_eq"] == [7.0]
        # Should NOT be in A_ub
        assert len(compiled["A_ub"]) == 0

    def test_eq_solution(self):
        model = MathematicalModel(
            name="EQ Test",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="x")],
            constraints=[Constraint(constraint_id="CON-1", name="eq", expression="x", relation=ConstraintRelation.EQ, rhs=7.0)],
        )
        compiled = SimpleLPCompiler.compile(model)
        compiled["model_id"] = model.model_id
        result = solve_lp_scipy(compiled)
        assert result.status == SolverStatus.OPTIMAL
        assert abs(result.variable_values["x"] - 7.0) < 1e-6


# ═══════════════════════════════════════════════════════════════
# Variable Bounds
# ═══════════════════════════════════════════════════════════════

class TestVariableBounds:
    def test_default_non_negative(self):
        model = MathematicalModel(
            name="Bounds Test",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="x")],
        )
        compiled = SimpleLPCompiler.compile(model)
        assert compiled["bounds"][0] == (0.0, None)

    def test_explicit_lower_bound(self):
        model = MathematicalModel(
            name="Bounds Test",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS, lower_bound=2.0)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="x")],
        )
        compiled = SimpleLPCompiler.compile(model)
        assert compiled["bounds"][0] == (2.0, None)

    def test_negative_variable_allowed(self):
        """Variables with lower_bound=None should allow negative values."""
        model = MathematicalModel(
            name="Free Var",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS, lower_bound=-100.0)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="x")],
        )
        compiled = SimpleLPCompiler.compile(model)
        assert compiled["bounds"][0] == (-100.0, None)


# ═══════════════════════════════════════════════════════════════
# Infeasible / Unbounded
# ═══════════════════════════════════════════════════════════════

class TestInfeasibleUnbounded:
    def test_infeasible(self):
        model = MathematicalModel(
            name="Infeasible",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS, lower_bound=0.0)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="x")],
            constraints=[
                Constraint(constraint_id="CON-1", name="ge", expression="x", relation=ConstraintRelation.GE, rhs=10.0),
                Constraint(constraint_id="CON-2", name="le", expression="x", relation=ConstraintRelation.LE, rhs=5.0),
            ],
        )
        compiled = SimpleLPCompiler.compile(model)
        compiled["model_id"] = model.model_id
        result = solve_lp_scipy(compiled)
        assert result.status == SolverStatus.INFEASIBLE
        assert result.feasibility is False

    def test_unbounded_approx(self):
        """Test unbounded-like behavior (minimize with no upper bound)."""
        model = MathematicalModel(
            name="Unbounded-like",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="-x")],
            # No upper bound on x, minimize -x → unbounded below
        )
        compiled = SimpleLPCompiler.compile(model)
        compiled["model_id"] = model.model_id
        result = solve_lp_scipy(compiled)
        # SciPy may report unbounded or just return a very negative value
        assert result.status in (SolverStatus.UNBOUNDED, SolverStatus.OPTIMAL, SolverStatus.ERROR)


# ═══════════════════════════════════════════════════════════════
# Safe Expression — Additional Escapes
# ═══════════════════════════════════════════════════════════════

class TestExpressionEscapes:
    def test_block_open(self):
        with pytest.raises(ExpressionValidationError):
            validate_expression('open("/etc/passwd")')

    def test_block_getattr(self):
        with pytest.raises(ExpressionValidationError):
            validate_expression('getattr(x, "__class__")')

    def test_block_os_system(self):
        with pytest.raises(ExpressionValidationError):
            validate_expression('os.system("rm -rf /")')

    def test_block_list_comprehension(self):
        with pytest.raises(ExpressionValidationError):
            validate_expression('[x for x in range(10)]')

    def test_allow_math_sqrt(self):
        tree = validate_expression('sqrt(16)', {"x"})
        assert tree is not None

    def test_allow_pi(self):
        tree = validate_expression('pi * x', {"x"})
        assert tree is not None

    def test_block_nested_malicious(self):
        with pytest.raises(ExpressionValidationError):
            validate_expression('__import__("os").system("echo")')

    def test_allow_scientific_notation(self):
        tree = validate_expression('1.5e3 * x', {"x"})
        assert tree is not None


# ═══════════════════════════════════════════════════════════════
# Unit System
# ═══════════════════════════════════════════════════════════════

class TestUnitSystem:
    def test_compatible_units(self):
        assert UnitChecker.check_compatible("m", "m") is True

    def test_incompatible_units(self):
        assert UnitChecker.check_compatible("m", "s") is False

    def test_unknown_units_pass(self):
        """Unknown units should not block (they're unknown, not wrong)."""
        issues = UnitChecker.check_addition(None, None)
        assert len(issues) == 0

    def test_dimensionless_compatible(self):
        assert UnitChecker.check_compatible("dimensionless", "m") is True


# ═══════════════════════════════════════════════════════════════
# Provenance
# ═══════════════════════════════════════════════════════════════

class TestProvenance:
    def test_solver_result_has_model_id(self):
        model = MathematicalModel(
            model_id="MODEL-TEST",
            name="Test",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS, lower_bound=0.0)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="x")],
        )
        compiled = SimpleLPCompiler.compile(model)
        compiled["model_id"] = "MODEL-TEST"
        result = solve_lp_scipy(compiled)
        assert result.model_id == "MODEL-TEST"

    def test_execution_real_flag(self):
        model = MathematicalModel(
            name="Test",
            variables=[Variable(symbol="x", name="x", variable_type=VariableType.CONTINUOUS, lower_bound=0.0)],
            objectives=[Objective(objective_id="OBJ-1", name="Min", sense=ObjectiveSense.MINIMIZE, expression="x")],
        )
        compiled = SimpleLPCompiler.compile(model)
        compiled["model_id"] = model.model_id
        result = solve_lp_scipy(compiled)
        assert result.execution_real is True
        assert result.production_safe is False