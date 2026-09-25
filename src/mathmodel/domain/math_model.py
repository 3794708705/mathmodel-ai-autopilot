"""MathModel AI — Mathematical domain models.

Typed schemas for variables, parameters, objectives, constraints,
equations, and the complete MathematicalModel.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════
# Symbol closure
# ═══════════════════════════════════════════════════════════════
#
# A model is only well-formed when every identifier an equation refers to is
# declared somewhere. The failure is detected here, but the *value* of the
# detection is the list of offending symbols: a caller that only receives the
# concatenated message cannot tell the next attempt which symbols to declare,
# so it regenerates the whole model and usually repeats the same mistake.

SYMBOL_CLOSURE_MARKER = "MathematicalModel validation failed: "

KIND_UNKNOWN_VARIABLE = "unknown_variable"
KIND_UNKNOWN_PARAMETER = "unknown_parameter"
KIND_UNKNOWN_DEPENDENCY = "unknown_dependency"
KIND_DUPLICATE_SYMBOL = "duplicate_symbol"
KIND_SYMBOL_COLLISION = "symbol_collision"


class SymbolClosureIssue(BaseModel):
    """One undeclared or conflicting symbol, with where it was referenced."""

    kind: str
    symbol: str
    location: str = ""


def describe_closure_issue(issue: SymbolClosureIssue) -> str:
    """Render an issue as the stable, human-readable line used in the message."""
    if issue.kind == KIND_UNKNOWN_VARIABLE:
        return f"Equation {issue.location}: unknown variable {issue.symbol}"
    if issue.kind == KIND_UNKNOWN_PARAMETER:
        return f"Equation {issue.location}: unknown parameter {issue.symbol}"
    if issue.kind == KIND_UNKNOWN_DEPENDENCY:
        return f"Equation {issue.location}: unknown dependency {issue.symbol}"
    if issue.kind == KIND_DUPLICATE_SYMBOL:
        return f"Duplicate variable symbol: {issue.symbol}"
    if issue.kind == KIND_SYMBOL_COLLISION:
        return f"Symbol collision between variable and parameter: {issue.symbol}"
    return f"{issue.kind}: {issue.symbol}"


def closure_issues(model: "MathematicalModel") -> list[SymbolClosureIssue]:
    """Every symbol an equation references that the model never declares."""
    var_ids = {v.variable_id for v in model.variables}
    param_ids = {p.parameter_id for p in model.parameters}
    eq_ids = {e.equation_id for e in model.equations}

    issues: list[SymbolClosureIssue] = []
    for eq in model.equations:
        for vid in eq.variable_ids:
            if vid not in var_ids:
                issues.append(SymbolClosureIssue(
                    kind=KIND_UNKNOWN_VARIABLE, symbol=vid,
                    location=str(eq.equation_id),
                ))
        for pid in eq.parameter_ids:
            if pid not in param_ids:
                issues.append(SymbolClosureIssue(
                    kind=KIND_UNKNOWN_PARAMETER, symbol=pid,
                    location=str(eq.equation_id),
                ))
        for dep_id in eq.dependencies:
            if dep_id not in eq_ids:
                issues.append(SymbolClosureIssue(
                    kind=KIND_UNKNOWN_DEPENDENCY, symbol=dep_id,
                    location=str(eq.equation_id),
                ))

    symbols = [v.symbol for v in model.variables]
    param_symbols = [p.symbol for p in model.parameters]
    for sym in dict.fromkeys(symbols):
        if symbols.count(sym) > 1:
            issues.append(SymbolClosureIssue(kind=KIND_DUPLICATE_SYMBOL, symbol=sym))
    for sym in dict.fromkeys(param_symbols):
        if sym in symbols:
            issues.append(SymbolClosureIssue(kind=KIND_SYMBOL_COLLISION, symbol=sym))

    return issues


def parse_closure_issues(message: str) -> list[SymbolClosureIssue]:
    """Recover the structured issues from a symbol-closure failure message.

    This is the exact inverse of `describe_closure_issue`. Pydantic wraps the
    ValueError raised by the model validator into a ValidationError and drops
    the exception object, so the message is the only channel that survives; the
    format and this parser are kept adjacent and round-trip tested together.
    """
    text = str(message)
    start = text.find(SYMBOL_CLOSURE_MARKER)
    if start < 0:
        return []
    body = text[start + len(SYMBOL_CLOSURE_MARKER):]

    # Pydantic appends its own error metadata to the end of the message
    # (` [type=value_error, input_value=..., input_type=dict]`), which would
    # otherwise be glued onto the last symbol in the list.
    metadata = body.find(" [type=")
    if metadata >= 0:
        body = body[:metadata]

    issues: list[SymbolClosureIssue] = []
    for raw in body.split(";"):
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"^Equation (?P<loc>.+?): unknown (variable|parameter|dependency) (?P<sym>.+)$", line)
        if match:
            kind = {
                "variable": KIND_UNKNOWN_VARIABLE,
                "parameter": KIND_UNKNOWN_PARAMETER,
                "dependency": KIND_UNKNOWN_DEPENDENCY,
            }[match.group(2)]
            issues.append(SymbolClosureIssue(
                kind=kind, symbol=match.group("sym").strip(),
                location=match.group("loc").strip(),
            ))
            continue
        if line.startswith("Duplicate variable symbol: "):
            issues.append(SymbolClosureIssue(
                kind=KIND_DUPLICATE_SYMBOL,
                symbol=line[len("Duplicate variable symbol: "):].strip(),
            ))
            continue
        if line.startswith("Symbol collision between variable and parameter: "):
            issues.append(SymbolClosureIssue(
                kind=KIND_SYMBOL_COLLISION,
                symbol=line[len("Symbol collision between variable and parameter: "):].strip(),
            ))
    return issues


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
        issues = closure_issues(self)
        if issues:
            raise ValueError(
                SYMBOL_CLOSURE_MARKER
                + "; ".join(describe_closure_issue(i) for i in issues)
            )
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