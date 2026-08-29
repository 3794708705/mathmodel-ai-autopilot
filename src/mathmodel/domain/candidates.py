"""MathModel AI — Model Candidate domain model.

Structured representation of candidate mathematical models
proposed by ModelExplorer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ModelFamily(str, Enum):
    """Broad family of mathematical model."""
    LINEAR_PROGRAMMING = "linear_programming"
    INTEGER_PROGRAMMING = "integer_programming"
    MIXED_INTEGER_PROGRAMMING = "mixed_integer_programming"
    NONLINEAR_PROGRAMMING = "nonlinear_programming"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    STOCHASTIC_PROGRAMMING = "stochastic_programming"
    GRAPH_THEORY = "graph_theory"
    NETWORK_FLOW = "network_flow"
    REGRESSION = "regression"
    CLASSIFICATION_ML = "classification_ml"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    SIMULATION = "simulation"
    QUEUEING = "queueing"
    GAME_THEORY = "game_theory"
    DECISION_TREES = "decision_trees"
    METAHEURISTIC = "metaheuristic"
    MULTI_OBJECTIVE = "multi_objective"
    HYBRID = "hybrid"
    OTHER = "other"


class ComponentRole(str, Enum):
    """Role of a model component in a chain."""
    INPUT_PROCESSING = "input_processing"
    FEATURE_ENGINEERING = "feature_engineering"
    CORE_MODEL = "core_model"
    POST_PROCESSING = "post_processing"
    EVALUATION = "evaluation"
    ENSEMBLE = "ensemble"


class ModelComponent(BaseModel):
    """A single component in a potentially multi-component model chain."""

    component_id: str = Field(default_factory=lambda: f"MC-{uuid4().hex[:8]}")
    name: str = Field(..., min_length=1)
    model_family: ModelFamily
    role: ComponentRole = ComponentRole.CORE_MODEL
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(
        default_factory=list,
        description="component_ids this depends on",
    )
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelCandidate(BaseModel):
    """A candidate mathematical model for a problem/subproblem.

    May contain a single model or a chain of components.
    No fabricated results — only the plan and structure.
    """

    candidate_id: str = Field(default_factory=lambda: f"CAND-{uuid4().hex[:8]}")
    name: str = Field(..., min_length=1)
    model_family: ModelFamily
    applicable_subproblems: list[str] = Field(
        default_factory=list,
        description="subproblem_ids this candidate addresses",
    )

    # Description
    summary: str = Field(..., min_length=1)
    mathematical_structure: str = Field(default="", description="Mathematical formulation overview")

    # Components
    components: list[ModelComponent] = Field(
        default_factory=list,
        description="Model components forming a chain",
    )

    # Requirements
    required_data: list[str] = Field(default_factory=list)
    required_assumptions: list[str] = Field(default_factory=list)

    # Assessment (structural, not empirical)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    # Plans (not results)
    implementation_plan: str = ""
    validation_plan: str = ""
    extension_options: list[str] = Field(default_factory=list)

    # Risk
    risk_flags: list[str] = Field(default_factory=list)

    # Meta
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_chain(self) -> bool:
        """Whether this candidate uses multiple components in a chain."""
        return len(self.components) > 1

    @property
    def core_components(self) -> list[ModelComponent]:
        """Return only CORE_MODEL components."""
        return [c for c in self.components if c.role == ComponentRole.CORE_MODEL]

    def similarity_key(self) -> str:
        """Generate a key for similarity comparison between candidates."""
        families = sorted(set(c.model_family.value for c in self.components))
        return f"{self.model_family.value}|{'/'.join(families)}"