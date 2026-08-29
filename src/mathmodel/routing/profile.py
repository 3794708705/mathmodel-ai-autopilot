"""MathModel AI — Task profile for model routing.

Describes the characteristics of a task so the router can
select the appropriate model tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    """Classification of task types for routing."""
    # Core modeling
    MATHEMATICAL_MODELING = "mathematical_modeling"
    PROBLEM_UNDERSTANDING = "problem_understanding"
    MODEL_EXPLORATION = "model_exploration"
    MODEL_JURY = "model_jury"

    # Data
    DATA_ANALYSIS = "data_analysis"
    FILE_PARSING = "file_parsing"

    # Code
    CODE_GENERATION = "code_generation"
    SOLVER_EXECUTION = "solver_execution"
    SANDBOX_SECURITY = "sandbox_security"

    # Verification
    VALIDATION = "validation"
    SENSITIVITY = "sensitivity"
    ROBUSTNESS = "robustness"
    RED_TEAM = "red_team"
    MODEL_REPAIR = "model_repair"

    # Paper
    PAPER_GENERATION = "paper_generation"
    CITATION_MANAGEMENT = "citation_management"
    CITATION_VERIFICATION = "citation_verification"
    VISUALIZATION = "visualization"

    # Infrastructure
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    CRUD = "crud"
    TESTING = "testing"


class ComplexityTier(str, Enum):
    """Estimated complexity of a task."""
    TRIVIAL = "trivial"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskProfile:
    """Profile describing a task's characteristics for model routing.

    The router uses this profile to select the appropriate model tier.
    """

    task_type: TaskType
    complexity: ComplexityTier = ComplexityTier.MEDIUM

    # Capability requirements
    reasoning_requirement: ComplexityTier = ComplexityTier.MEDIUM
    math_requirement: ComplexityTier = ComplexityTier.LOW
    coding_requirement: ComplexityTier = ComplexityTier.LOW
    multimodal_requirement: ComplexityTier = ComplexityTier.LOW
    long_context_requirement: ComplexityTier = ComplexityTier.LOW
    review_requirement: ComplexityTier = ComplexityTier.LOW

    # Risk assessment
    security_risk: ComplexityTier = ComplexityTier.LOW
    blast_radius: ComplexityTier = ComplexityTier.LOW
    cost_sensitivity: ComplexityTier = ComplexityTier.MEDIUM
    deadline_pressure: ComplexityTier = ComplexityTier.LOW

    # Runtime state
    retry_count: int = 0

    # Metadata
    metadata: dict = field(default_factory=dict)

    @classmethod
    def for_task_type(cls, task_type: TaskType, **overrides) -> "TaskProfile":
        """Create a TaskProfile with sensible defaults for a task type."""
        defaults: dict = {
            "task_type": task_type,
        }

        # Set reasonable defaults based on task type
        if task_type == TaskType.MATHEMATICAL_MODELING:
            defaults.update(
                complexity=ComplexityTier.CRITICAL,
                reasoning_requirement=ComplexityTier.CRITICAL,
                math_requirement=ComplexityTier.CRITICAL,
                blast_radius=ComplexityTier.CRITICAL,
            )
        elif task_type == TaskType.SANDBOX_SECURITY:
            defaults.update(
                complexity=ComplexityTier.CRITICAL,
                security_risk=ComplexityTier.CRITICAL,
                blast_radius=ComplexityTier.CRITICAL,
            )
        elif task_type == TaskType.CODE_GENERATION:
            defaults.update(
                coding_requirement=ComplexityTier.HIGH,
                blast_radius=ComplexityTier.HIGH,
            )
        elif task_type == TaskType.DOCUMENTATION:
            defaults.update(
                complexity=ComplexityTier.LOW,
                reasoning_requirement=ComplexityTier.LOW,
            )

        defaults.update(overrides)
        return cls(**defaults)