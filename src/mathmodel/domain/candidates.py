"""MathModel AI — Model Candidate domain model.

Structured representation of candidate mathematical models
proposed by ModelExplorer.
"""

from __future__ import annotations

from collections import defaultdict, deque
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
    def has_cycles(self) -> bool:
        """Check if the component dependency graph has cycles."""
        if not self.components:
            return False
        return bool(self._find_cycles())

    def _find_cycles(self) -> list[list[str]]:
        """Find cycles in the component dependency graph using topological sort."""
        comp_ids = {c.component_id for c in self.components}
        # Build adjacency list
        graph: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {cid: 0 for cid in comp_ids}

        for c in self.components:
            for dep in c.dependencies:
                if dep in comp_ids:
                    graph[dep].append(c.component_id)
                    in_degree[c.component_id] = in_degree.get(c.component_id, 0) + 1

        # Kahn's algorithm for topological sort
        queue = deque([cid for cid in comp_ids if in_degree.get(cid, 0) == 0])
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If visited != len(comp_ids), there are cycles
        if visited != len(comp_ids):
            remaining = [cid for cid in comp_ids if in_degree.get(cid, 0) > 0]
            return [remaining]  # Return the cycle participants
        return []

    @property
    def is_dag(self) -> bool:
        """Whether the component graph is a valid DAG (no cycles)."""
        return not self.has_cycles

    @property
    def core_components(self) -> list[ModelComponent]:
        """Return only CORE_MODEL components."""
        return [c for c in self.components if c.role == ComponentRole.CORE_MODEL]

    def similarity_key(self) -> str:
        """Generate a key for similarity comparison between candidates.

        Uses model_family, component families, and mathematical_structure
        to detect genuinely similar candidates.
        """
        families = sorted(set(c.model_family.value for c in self.components))
        # Normalize mathematical structure for comparison
        math_key = (
            self.mathematical_structure.lower()
            .replace(" ", "")
            .replace("\n", "")[:80]
        )
        return f"{self.model_family.value}|{'/'.join(families)}|{math_key}"

    def is_essentially_same_as(self, other: "ModelCandidate") -> bool:
        """Check if two candidates are essentially the same model.

        Compares name similarity, component structure, and assumptions
        — not just model_family.
        """
        if self.candidate_id == other.candidate_id:
            return True

        # Same model_family
        same_family = self.model_family == other.model_family

        # Name similarity (normalized)
        self_name = self.name.lower().replace(" ", "").replace("-", "").replace("_", "")
        other_name = other.name.lower().replace(" ", "").replace("-", "").replace("_", "")
        names_similar = self_name == other_name

        # Component structure similarity
        self_comp_families = sorted(c.model_family.value for c in self.components)
        other_comp_families = sorted(c.model_family.value for c in other.components)
        same_components = self_comp_families == other_comp_families

        # Same mathematical structure (normalized)
        self_math = self.mathematical_structure.lower().replace(" ", "").replace("\n", "")
        other_math = other.mathematical_structure.lower().replace(" ", "").replace("\n", "")
        same_math = self_math == other_math and len(self_math) > 0

        # Same assumptions
        self_assumptions = sorted(a.lower().strip() for a in self.required_assumptions)
        other_assumptions = sorted(a.lower().strip() for a in other.required_assumptions)
        same_assumptions = (
            self_assumptions == other_assumptions
            and len(self_assumptions) > 0
        )

        # Two candidates are "essentially same" if:
        # - same family AND similar names, OR
        # - same family AND same math structure, OR
        # - same family AND same components AND same assumptions
        return (
            (same_family and names_similar)
            or (same_family and same_math)
            or (same_family and same_components and same_assumptions)
        )