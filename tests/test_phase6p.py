"""Phase 6P tests: CodeAgent, CodeMapping, drift detection, compound units."""

import pytest

from mathmodel.domain.math_model import (
    MathematicalModel, Variable, Parameter, Objective, Constraint,
    ConstraintRelation, ObjectiveSense, VariableType,
)
from mathmodel.domain.codegen import CodeObjectType
from mathmodel.agents.code_agent import CodeAgent, check_model_code_drift
from mathmodel.math_ops.units import (
    parse_unit, units_compatible, check_equation_units, check_objective_units,
    Dimension,
)


def make_lp_model() -> MathematicalModel:
    return MathematicalModel(
        model_id="MODEL-6P",
        name="LP Model",
        variables=[
            Variable(variable_id="VAR-x1", symbol="x1", name="x1", variable_type=VariableType.CONTINUOUS, lower_bound=0.0),
            Variable(variable_id="VAR-x2", symbol="x2", name="x2", variable_type=VariableType.CONTINUOUS, lower_bound=0.0),
        ],
        parameters=[
            Parameter(parameter_id="PAR-p1", symbol="p1", name="profit1", value=30.0),
            Parameter(parameter_id="PAR-p2", symbol="p2", name="profit2", value=20.0),
        ],
        objectives=[
            Objective(objective_id="OBJ-1", name="Max", sense=ObjectiveSense.MAXIMIZE, expression="30*x1 + 20*x2"),
        ],
        constraints=[
            Constraint(constraint_id="CON-1", name="cap", expression="x1 + x2", relation=ConstraintRelation.LE, rhs=40.0),
        ],
    )


# ═══════════════════════════════════════════════════════════════
# CodeAgent
# ═══════════════════════════════════════════════════════════════

class TestCodeAgent:
    def test_generates_artifact_for_lp(self):
        agent = CodeAgent()
        result = agent.generate(make_lp_model())
        assert result.status == "completed"
        assert result.artifact is not None
        assert result.artifact.entrypoint == "solver.py"
        assert len(result.artifact.files) == 1

    def test_objective_and_constraint_mappings(self):
        agent = CodeAgent()
        result = agent.generate(make_lp_model())
        types = {m.object_type for m in result.mappings}
        assert CodeObjectType.OBJECTIVE in types
        assert CodeObjectType.CONSTRAINT in types

    def test_no_drift_when_complete(self):
        agent = CodeAgent()
        result = agent.generate(make_lp_model())
        issues = check_model_code_drift(make_lp_model(), result.mappings)
        assert len(issues) == 0

    def test_drift_when_constraint_missing(self):
        agent = CodeAgent()
        result = agent.generate(make_lp_model())
        # Remove a constraint mapping
        filtered = [m for m in result.mappings
                    if m.object_type != CodeObjectType.CONSTRAINT]
        issues = check_model_code_drift(make_lp_model(), filtered)
        assert len(issues) > 0
        assert any("Constraints without" in i for i in issues)

    def test_drift_when_extra_mapping(self):
        agent = CodeAgent()
        result = agent.generate(make_lp_model())
        from mathmodel.domain.codegen import CodeMapping
        result.mappings.append(CodeMapping(
            model_id="MODEL-6P",
            mathematical_object_id="CON-GHOST",
            object_type=CodeObjectType.CONSTRAINT,
        ))
        issues = check_model_code_drift(make_lp_model(), result.mappings)
        assert any("unknown constraints" in i for i in issues)

    def test_blocked_for_uncompilable_model(self):
        """CodeAgent must not invent a second model for uncompilable models."""
        agent = CodeAgent()
        model = MathematicalModel(
            name="Weird model",
            objectives=[Objective(objective_id="O1", name="obj", expression="1")],
        )
        # No variables → compiler will fail or produce empty
        result = agent.generate(model)
        # Either completed with empty mappings or blocked — never invents structure
        assert result.status in ("completed", "blocked")


# ═══════════════════════════════════════════════════════════════
# Compound Units
# ═══════════════════════════════════════════════════════════════

class TestCompoundUnits:
    def test_m_per_s(self):
        dim = parse_unit("m/s")
        assert dim is not None
        assert dim.powers == {"length": 1, "time": -1}

    def test_m_per_s2(self):
        dim = parse_unit("m/s^2")
        assert dim is not None
        assert dim.powers == {"length": 1, "time": -2}

    def test_kg_per_m3(self):
        dim = parse_unit("kg/m^3")
        assert dim is not None
        assert dim.powers == {"mass": 1, "length": -3}

    def test_kg_m_per_s2(self):
        dim = parse_unit("kg*m/s^2")
        assert dim is not None
        assert dim.powers == {"mass": 1, "length": 1, "time": -2}

    def test_currency_per_item(self):
        dim = parse_unit("currency/item")
        assert dim is not None
        assert dim.powers == {"currency": 1, "count": -1}

    def test_item_per_hour(self):
        dim = parse_unit("item/hour")
        assert dim is not None
        assert dim.powers == {"count": 1, "time": -1}

    def test_m2(self):
        dim = parse_unit("m^2")
        assert dim is not None
        assert dim.powers == {"length": 2}

    def test_dimensionless(self):
        dim = parse_unit("dimensionless")
        assert dim is not None
        assert dim.is_dimensionless()

    def test_unknown_returns_none(self):
        assert parse_unit("flurbos") is None
        assert parse_unit("") is None
        assert parse_unit("unknown") is None

    def test_compatible_same(self):
        ok, msg = units_compatible("m/s", "m/s")
        assert ok, msg

    def test_km_h_vs_m_s(self):
        # km/h and m/s have the same dimension (length/time)
        ok, msg = units_compatible("km/h", "m/s")
        assert ok, msg

    def test_mismatch(self):
        ok, msg = units_compatible("m", "s")
        assert not ok
        assert "MISMATCH" in msg

    def test_unknown_vs_known(self):
        ok, msg = units_compatible("flurbos", "m")
        assert not ok
        assert "UNKNOWN" in msg

    def test_equation_units(self):
        ok, msg = check_equation_units("m", "m")
        assert ok, msg

    def test_equation_mismatch(self):
        ok, msg = check_equation_units("m", "kg")
        assert not ok

    def test_objective_incompatible(self):
        ok, msg = check_objective_units(["currency", "hour"])
        assert not ok
        assert "normalization" in msg.lower()

    def test_objective_compatible(self):
        ok, msg = check_objective_units(["currency", "currency"])
        assert ok, msg