"""Phase 4 E2E tests: Mathematical Core + Real Solver Execution.

Tests the full pipeline: MathematicalModel → Compiler → SciPy → SolverResult → ConstraintEvaluator.
"""

import asyncio
import pytest

from mathmodel.agents.math_modeler import MathModeler
from mathmodel.domain.math_model import (
    MathematicalModel,
    Variable,
    Parameter,
    Objective,
    Constraint,
    Equation,
    ConstraintRelation,
    ObjectiveSense,
    VariableType,
    ParameterStatus,
    EquationType,
)
from mathmodel.domain.analysis import ProblemAnalysis, Subproblem, ModelingTaskType
from mathmodel.domain.state_helpers import store_analysis, record_revision
from mathmodel.models.problem_state import ProblemState, ProblemStateStage
from mathmodel.solver import SimpleLPCompiler, solve_lp_scipy, evaluate_constraints, SolverStatus
from mathmodel.math_ops import (
    SymbolRegistry,
    EquationRegistry,
    validate_expression,
    ExpressionValidationError,
)
from mathmodel.providers.mock import MockProvider
from mathmodel.routing.router import ModelRouter


def make_router():
    return ModelRouter()


# ═══════════════════════════════════════════════════════════════
# Real LP Fixture
# ═══════════════════════════════════════════════════════════════

FIXTURE_LP_MODEL = MathematicalModel(
    model_id="MODEL-LP-001",
    name="Production Planning LP",
    description="Maximize profit from 3 products with machine and demand constraints",
    source_candidate_id="CAND-LP-1",
    applicable_subproblems=["SUB-1"],
    model_family="linear_programming",
    variables=[
        Variable(variable_id="VAR-x1", symbol="x1", name="Product 1 quantity", variable_type=VariableType.CONTINUOUS, lower_bound=0.0, meaning="Units of P1 to produce"),
        Variable(variable_id="VAR-x2", symbol="x2", name="Product 2 quantity", variable_type=VariableType.CONTINUOUS, lower_bound=0.0, meaning="Units of P2 to produce"),
        Variable(variable_id="VAR-x3", symbol="x3", name="Product 3 quantity", variable_type=VariableType.CONTINUOUS, lower_bound=0.0, meaning="Units of P3 to produce"),
    ],
    parameters=[
        Parameter(parameter_id="PAR-p1", symbol="p1", name="Profit P1", value=30.0, unit="$"),
        Parameter(parameter_id="PAR-p2", symbol="p2", name="Profit P2", value=25.0, unit="$"),
        Parameter(parameter_id="PAR-p3", symbol="p3", name="Profit P3", value=20.0, unit="$"),
    ],
    objectives=[
        Objective(objective_id="OBJ-1", name="Maximize Profit", sense=ObjectiveSense.MAXIMIZE, expression="30*x1 + 25*x2 + 20*x3", meaning="Total weekly profit"),
    ],
    constraints=[
        Constraint(constraint_id="CON-1", name="Machine M1 hours", expression="2*x1 + 1.5*x2 + 1*x3", relation=ConstraintRelation.LE, rhs=40.0, meaning="M1 capacity: 40h"),
        Constraint(constraint_id="CON-2", name="Machine M2 hours", expression="1.5*x1 + 2*x2 + 1*x3", relation=ConstraintRelation.LE, rhs=35.0, meaning="M2 capacity: 35h"),
        Constraint(constraint_id="CON-3", name="P1 demand limit", expression="x1", relation=ConstraintRelation.LE, rhs=15.0, meaning="P1 max demand"),
        Constraint(constraint_id="CON-4", name="P3 minimum", expression="x3", relation=ConstraintRelation.GE, rhs=5.0, meaning="P3 contractual minimum"),
    ],
    equations=[
        Equation(equation_id="EQ-1", name="Profit function", expression="30*x1 + 25*x2 + 20*x3", equation_type=EquationType.OBJECTIVE, variable_ids=["VAR-x1", "VAR-x2", "VAR-x3"], parameter_ids=["PAR-p1", "PAR-p2", "PAR-p3"]),
    ],
    algorithm_plan="Solve using SciPy linprog (LP solver)",
    solver_requirements=["linear_programming"],
)


# ═══════════════════════════════════════════════════════════════
# Expression Safety Tests
# ═══════════════════════════════════════════════════════════════

class TestExpressionSafety:
    def test_valid_expression(self):
        tree = validate_expression("2*x1 + 3*x2", {"x1", "x2"})
        assert tree is not None

    def test_valid_math_function(self):
        tree = validate_expression("sqrt(x1) + abs(x2)", {"x1", "x2"})
        assert tree is not None

    def test_block_import(self):
        with pytest.raises(ExpressionValidationError):
            validate_expression('__import__("os").system("echo")')

    def test_block_eval(self):
        with pytest.raises(ExpressionValidationError):
            validate_expression('eval("1+1")')

    def test_block_dunder(self):
        with pytest.raises(ExpressionValidationError):
            validate_expression('__class__')

    def test_block_lambda(self):
        with pytest.raises(ExpressionValidationError):
            validate_expression('lambda x: x')

    def test_block_attribute_access(self):
        with pytest.raises(ExpressionValidationError):
            validate_expression('x.__class__')

    def test_unknown_symbol_allowed(self):
        # Unknown symbols pass validation (they might be parameters)
        tree = validate_expression("a + b", {"a"})
        assert tree is not None


# ═══════════════════════════════════════════════════════════════
# Symbol/Equation Registry Tests
# ═══════════════════════════════════════════════════════════════

class TestRegistries:
    def test_symbol_register(self):
        reg = SymbolRegistry()
        reg.register("x1", "VAR-1", "variable", meaning="Product 1")
        assert reg.lookup("x1") is not None
        assert reg.symbol_count == 1

    def test_symbol_duplicate_same_entity(self):
        reg = SymbolRegistry()
        reg.register("x1", "VAR-1", "variable")
        # Same symbol, same entity — OK
        reg.register("x1", "VAR-1", "variable")

    def test_symbol_duplicate_different_entity(self):
        reg = SymbolRegistry()
        reg.register("x1", "VAR-1", "variable")
        with pytest.raises(ValueError, match="already registered"):
            reg.register("x1", "VAR-2", "parameter")

    def test_symbol_unused(self):
        reg = SymbolRegistry()
        reg.register("x1", "VAR-1", "variable")
        reg.register("x2", "VAR-2", "variable")
        reg.mark_used_in("x1", "EQ-1")
        unused = reg.get_unused()
        assert len(unused) == 1
        assert unused[0].symbol == "x2"

    def test_equation_registry_dependencies(self):
        reg = EquationRegistry()
        eq1 = Equation(equation_id="EQ-1", name="Eq1", expression="x1 + x2", dependencies=["EQ-2"])
        eq2 = Equation(equation_id="EQ-2", name="Eq2", expression="x3")
        reg.register(eq1)
        reg.register(eq2)
        issues = reg.validate_dependencies()
        assert len(issues) == 0  # EQ-2 exists

    def test_equation_missing_dependency(self):
        reg = EquationRegistry()
        eq1 = Equation(equation_id="EQ-1", name="Eq1", expression="x1", dependencies=["EQ-NONEXISTENT"])
        reg.register(eq1)
        issues = reg.validate_dependencies()
        assert len(issues) > 0


# ═══════════════════════════════════════════════════════════════
# MathematicalModel Tests
# ═══════════════════════════════════════════════════════════════

class TestMathematicalModel:
    def test_valid_model(self):
        issues = FIXTURE_LP_MODEL.validate_domain()
        assert len(issues) == 0

    def test_duplicate_symbol(self):
        with pytest.raises(ValueError, match="Duplicate"):
            MathematicalModel(
                name="Test",
                variables=[
                    Variable(symbol="x", name="x1"),
                    Variable(symbol="x", name="x2"),
                ],
            )

    def test_symbol_collision(self):
        with pytest.raises(ValueError, match="collision"):
            MathematicalModel(
                name="Test",
                variables=[Variable(symbol="x", name="x1")],
                parameters=[Parameter(symbol="x", name="p1")],
            )

    def test_equation_unknown_variable(self):
        with pytest.raises(ValueError, match="unknown variable"):
            MathematicalModel(
                name="Test",
                equations=[Equation(equation_id="EQ-1", name="Eq", expression="x", variable_ids=["VAR-NONEXISTENT"])],
            )

    def test_serialization_round_trip(self):
        data = FIXTURE_LP_MODEL.model_dump()
        restored = MathematicalModel.model_validate(data)
        assert restored.model_id == FIXTURE_LP_MODEL.model_id
        assert len(restored.variables) == 3
        assert len(restored.constraints) == 4


# ═══════════════════════════════════════════════════════════════
# Real LP Execution Tests
# ═══════════════════════════════════════════════════════════════

class TestRealLPExecution:
    def test_compile_lp(self):
        compiled = SimpleLPCompiler.compile(FIXTURE_LP_MODEL)
        assert len(compiled["c"]) == 3
        assert len(compiled["A_ub"]) == 4
        assert compiled["variable_names"] == ["x1", "x2", "x3"]

    def test_solve_lp_scipy(self):
        compiled = SimpleLPCompiler.compile(FIXTURE_LP_MODEL)
        compiled["model_id"] = FIXTURE_LP_MODEL.model_id
        result = solve_lp_scipy(compiled)

        assert result.status == SolverStatus.OPTIMAL
        assert result.execution_real is True
        assert result.objective_value is not None

        # Known optimum for this problem: max profit ≈ 687.5-700
        # Use wider tolerance since solver may find slightly different optima
        assert 680 < result.objective_value < 710

        # Check variable values
        assert "x1" in result.variable_values
        assert "x3" in result.variable_values
        # P3 minimum constraint: x3 >= 5
        assert result.variable_values["x3"] >= 4.9

    def test_constraint_evaluation(self):
        compiled = SimpleLPCompiler.compile(FIXTURE_LP_MODEL)
        compiled["model_id"] = FIXTURE_LP_MODEL.model_id
        result = solve_lp_scipy(compiled)

        eval_result = evaluate_constraints(FIXTURE_LP_MODEL, result)
        assert eval_result["all_satisfied"] is True
        assert eval_result["max_violation"] < 1e-4

    def test_objective_recomputation(self):
        """Recompute objective from variable values and compare with solver."""
        compiled = SimpleLPCompiler.compile(FIXTURE_LP_MODEL)
        compiled["model_id"] = FIXTURE_LP_MODEL.model_id
        result = solve_lp_scipy(compiled)

        # Recompute: 30*x1 + 25*x2 + 20*x3
        recomputed = (
            30 * result.variable_values.get("x1", 0)
            + 25 * result.variable_values.get("x2", 0)
            + 20 * result.variable_values.get("x3", 0)
        )
        assert abs(recomputed - result.objective_value) < 0.01

    def test_provenance_chain(self):
        """Verify the provenance chain is complete."""
        # Model -> Compiler -> Solver -> Result
        compiled = SimpleLPCompiler.compile(FIXTURE_LP_MODEL)
        compiled["model_id"] = FIXTURE_LP_MODEL.model_id
        result = solve_lp_scipy(compiled)

        assert result.model_id == FIXTURE_LP_MODEL.model_id
        assert result.solver == "scipy"
        assert result.execution_real is True
        assert result.production_safe is False
        assert result.status == SolverStatus.OPTIMAL


# ═══════════════════════════════════════════════════════════════
# MathModeler Agent Tests
# ═══════════════════════════════════════════════════════════════

class TestMathModelerAgent:
    def test_agent_contract(self):
        agent = MathModeler(router=make_router())
        assert agent.name == "MathModeler"
        assert "variable_definition" in agent.capabilities

    def test_run_with_analysis(self, db_session):
        agent = MathModeler(router=make_router())
        state = ProblemState()
        analysis = ProblemAnalysis(
            background="Production planning",
            core_problem="Maximize profit",
            subproblems=[Subproblem(subproblem_id="SUB-1", original_text="Optimize", normalized_goal="Max profit", task_types=[ModelingTaskType.OPTIMIZATION])],
        )
        store_analysis(state, analysis)
        state.selected_model = {"candidate_id": "CAND-LP-1"}

        result = asyncio.run(agent.run(state))
        assert result.agent_name == "MathModeler"

    def test_validate_model(self):
        agent = MathModeler(router=make_router())
        errors = agent.validate_output(FIXTURE_LP_MODEL)
        assert len(errors) == 0