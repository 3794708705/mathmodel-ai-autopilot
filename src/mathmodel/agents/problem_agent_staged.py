"""MathModel AI — Staged ProblemAgent decomposition.

Small, independent structured outputs assembled deterministically
into a full ProblemAnalysis. Avoids monolithic schema failures.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# Pass A — Core Problem Understanding
# ═══════════════════════════════════════════════════════════════

class PassAOutput(BaseModel):
    concise_summary: str = Field(default="", description="1-2 sentence problem summary")
    objective: list[str] = Field(default_factory=list, description="Primary objectives, max 3")
    task_types: list[str] = Field(default_factory=list, description="Modeling task types, max 5")
    subproblems: list[str] = Field(default_factory=list, description="Subproblem descriptions, max 5")


# ═══════════════════════════════════════════════════════════════
# Pass B — Facts & Data
# ═══════════════════════════════════════════════════════════════

class FactItem(BaseModel):
    text: str = Field(default="", description="Fact statement")
    is_numerical: bool = False
    source: str = Field(default="problem_statement", description="Where this fact comes from")


class DataItem(BaseModel):
    text: str = Field(default="", description="Data reference")
    file_reference: str = Field(default="", description="Attachment file name if applicable")


class PassBOutput(BaseModel):
    facts: list[FactItem] = Field(default_factory=list, description="Facts from problem, max 12")
    data_refs: list[DataItem] = Field(default_factory=list, description="Data references, max 8")


# ═══════════════════════════════════════════════════════════════
# Pass C — Assumptions & Ambiguities
# ═══════════════════════════════════════════════════════════════

class AmbiguityItem(BaseModel):
    text: str = Field(default="", description="Ambiguous aspect")
    suggested_resolution: str = Field(default="", description="Proposed assumption to resolve")


class PassCOutput(BaseModel):
    ambiguities: list[AmbiguityItem] = Field(default_factory=list, description="Ambiguities, max 5")
    proposed_assumptions: list[str] = Field(default_factory=list, description="Proposed assumptions, max 6")


# ═══════════════════════════════════════════════════════════════
# Pass D — Modeling Requirements
# ═══════════════════════════════════════════════════════════════

class ConstraintItem(BaseModel):
    text: str = Field(default="", description="Constraint description")
    is_explicit: bool = True


class PassDOutput(BaseModel):
    explicit_constraints: list[str] = Field(default_factory=list, description="Explicit constraints, max 10")
    implicit_conditions: list[str] = Field(default_factory=list, description="Implicit conditions, max 5")
    evaluation_criteria: list[str] = Field(default_factory=list, description="How to evaluate solution, max 3")