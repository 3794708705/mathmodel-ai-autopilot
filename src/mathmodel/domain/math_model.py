"""MathModel AI — Mathematical domain models.

Typed schemas for variables, parameters, objectives, constraints,
equations, and the complete MathematicalModel.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════
# Variable
# ═══════════════════════════════════════════════════════════════

class VariableType(str, Enum):
    DECISION = "decision"
    STATE = "state"
    AUXILIARY = "auxiliary"
    RANDOM = "random"
    DERIVED = "derived"
    BINARY = "binary"
    INTEGER = "integer"
    CONTINUOUS = "continuous"


class Variable(BaseModel):
    """A mathematical variable in the model."""

    variable_id: str = Field(default_factory=lambda: f"VAR-{uuid4().hex[:8]}")
    symbol: str = Field(..., min_length=1, description="Mathematical symbol, e.g. x_1")
    name: str = Field(..., min_length=1)
    meaning: str = Field(default="")
    variable_type: VariableType = VariableType.CONTINUOUS
    domain: str = Field(default="R", description="Domain of the variable, e.g. R, Z, {0,1}")
    unit: Optional[str] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    indices: list[str] = Field(default_factory=list, description="e.g. ['i', 'j'] for x_{i,j}")
    dimensions: Optional[int] = None
    source: str = Field(default="", description="Where this variable comes from")
    first_used_in: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Parameter
# ═══════════════════════════════════════════════════════════════

class ParameterStatus(str, Enum):
    KNOWN = "known"
    ESTIMATED = "estimated"
    REQUIRED = "required"
    CALIBRATED = "calibrated"


class Parameter(BaseModel):
    """A parameter in the mathematical model."""

    parameter_id: str = Field(default_factory=lambda: f"PAR-{uuid4().hex[:8]}")
    symbol: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    meaning: str = Field(default="")
    value: Optional[float] = None
    unit: Optional[str] = None
    source: str = Field(default="")
    source_reference: Optional[str] = None
    estimated: bool = False
    estimation_method: Optional[str] = None
    uncertainty: Optional[float] = None
    status: ParameterStatus = ParameterStatus.KNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Objective
# ═══════════════════════════════════════════════════════════════

class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Objective(BaseModel):
    """An objective function in the model."""

    objective_id: str = Field(default_factory=lambda: f"OBJ-{uuid4().hex[:8]}")
    name: str = Field(..., min_length=1)
    sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    expression: str = Field(..., min_length=1, description="Mathematical expression")
    meaning: str = Field(default="")
    priority: int = Field(default=1, ge=1)
    weight: float = Field(default=1.0)
    normalization: Optional[str] = None
    unit: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Constraint
# ═══════════════════════════════════════════════════════════════

class ConstraintRelation(str, Enum):
    EQ = "eq"
    LE = "le"
    GE = "ge"


class Constraint(BaseModel):
    """A constraint in the mathematical model."""

    constraint_id: str = Field(default_factory=lambda: f"CON-{uuid4().hex[:8]}")
    name: str = Field(..., min_length=1)
    expression: str = Field(..., min_length=1)
    relation: ConstraintRelation = ConstraintRelation.LE
    rhs: float = 0.0
    meaning: str = Field(default="")
    source: str = Field(default="")
    hard_or_soft: str = Field(default="hard")
    penalty: Optional[float] = None
    tolerance: float = Field(default=1e-6, ge=0.0)
    unit: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Equation
# ═══════════════════════════════════════════════════════════════

class EquationType(str, Enum):
    DEFINITION = "definition"
    OBJECTIVE = "objective"
    CONSTRAINT = "constraint"
    BALANCE = "balance"
    TRANSITION = "transition"
    PROBABILITY = "probability"
    STATISTICAL = "statistical"
    DERIVED = "derived"
    BOUNDARY = "boundary"


class Equation(BaseModel):
    """A formal equation in the mathematical model."""

    equation_id: str = Field(default_factory=lambda: f"EQ-{uuid4().hex[:8]}")
    name: str = Field(..., min_length=1)
    expression: str = Field(..., min_length=1)
    latex: str = Field(default="")
    equation_type: EquationType = EquationType.DEFINITION
    meaning: str = Field(default="")
    derivation: str = Field(default="")
    source_conditions: list[str] = Field(default_factory=list)
    variable_ids: list[str] = Field(default_factory=list)
    parameter_ids: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list, description="equation_ids")
    unit_check: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# MathematicalModel
# ═══════════════════════════════════════════════════════════════

class MathematicalModel(BaseModel):
    """Complete mathematical model for a problem/subproblem.

    This is the SOURCE OF TRUTH for all subsequent computation.
    Not the prompt. Not the generated code. Not the solver output.
    """

    model_id: str = Field(default_factory=lambda: f"MODEL-{uuid4().hex[:8]}")
    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    source_candidate_id: Optional[str] = None
    applicable_subproblems: list[str] = Field(default_factory=list)
    model_family: Optional[str] = None

    # Components
    assumptions: list[str] = Field(default_factory=list, description="Accepted assumption evidence IDs")
    variables: list[Variable] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    equations: list[Equation] = Field(default_factory=list)
    objectives: list[Objective] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    boundary_conditions: list[str] = Field(default_factory=list)

    # Plans
    model_components: list[str] = Field(default_factory=list)
    algorithm_plan: str = Field(default="")
    solver_requirements: list[str] = Field(default_factory=list)
    validation_plan: str = Field(default="")

    # Meta
    version: int = 1
    created_by: str = Field(default="MathModeler")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> "MathematicalModel":
        """Validate internal references."""
        var_ids = {v.variable_id for v in self.variables}
        param_ids = {p.parameter_id for p in self.parameters}
        eq_ids = {e.equation_id for e in self.equations}

        errors = []

        # Equation references
        for eq in self.equations:
            for vid in eq.variable_ids:
                if vid not in var_ids:
                    errors.append(f"Equation {eq.equation_id}: unknown variable {vid}")
            for pid in eq.parameter_ids:
                if pid not in param_ids:
                    errors.append(f"Equation {eq.equation_id}: unknown parameter {pid}")
            for dep_id in eq.dependencies:
                if dep_id not in eq_ids:
                    errors.append(f"Equation {eq.equation_id}: unknown dependency {dep_id}")

        # Symbol uniqueness
        symbols = [v.symbol for v in self.variables]
        param_symbols = [p.symbol for p in self.parameters]
        for sym in set(symbols):
            if symbols.count(sym) > 1:
                errors.append(f"Duplicate variable symbol: {sym}")
        for sym in param_symbols:
            if symbols.count(sym) > 0:
                errors.append(f"Symbol collision between variable and parameter: {sym}")

        if errors:
            raise ValueError("MathematicalModel validation failed: " + "; ".join(errors))

        return self

    def validate_domain(self) -> list[str]:
        """Validate all domain invariants."""
        issues = []

        # Must have at least one objective
        if not self.objectives:
            issues.append("Model must have at least one objective")

        # All required parameters must have values
        for p in self.parameters:
            if p.status == ParameterStatus.REQUIRED and p.value is None:
                issues.append(f"Parameter {p.parameter_id} ({p.symbol}) is REQUIRED but has no value")
            if p.status == ParameterStatus.ESTIMATED and p.uncertainty is None:
                issues.append(f"Parameter {p.parameter_id} ({p.symbol}) is ESTIMATED but has no uncertainty")

        return issues