"""MathModel AI — Domain schemas.

Typed Pydantic models for the Reasoning Core.
All domain objects pass through these schemas before persistence.
"""

from mathmodel.domain.evidence import EvidenceItem, EvidenceType, EvidenceStatus
from mathmodel.domain.analysis import (
    ProblemAnalysis,
    Subproblem,
    Ambiguity,
    AmbiguityImpact,
    ModelingTaskType,
)
from mathmodel.domain.candidates import (
    ModelCandidate,
    ModelComponent,
    ModelFamily,
    ComponentRole,
)
from mathmodel.domain.eligibility import (
    EligibilityResult,
    EligibilityPolicy,
    EligibilityFailure,
    EligibilityWarning,
    EligibilityCheck,
)
from mathmodel.domain.jury import (
    JuryScore,
    JuryDimension,
    ModelJuryResult,
    SelectionOverride,
    DEFAULT_JURY_WEIGHTS,
)
from mathmodel.domain.literature import (
    LiteratureQueryPlan,
    LiteratureQuery,
    LiteratureQueryStatus,
)

__all__ = [
    # Evidence
    "EvidenceItem",
    "EvidenceType",
    "EvidenceStatus",
    # Analysis
    "ProblemAnalysis",
    "Subproblem",
    "Ambiguity",
    "AmbiguityImpact",
    "ModelingTaskType",
    # Candidates
    "ModelCandidate",
    "ModelComponent",
    "ModelFamily",
    "ComponentRole",
    # Eligibility
    "EligibilityResult",
    "EligibilityPolicy",
    "EligibilityFailure",
    "EligibilityWarning",
    "EligibilityCheck",
    # Jury
    "JuryScore",
    "JuryDimension",
    "ModelJuryResult",
    "SelectionOverride",
    "DEFAULT_JURY_WEIGHTS",
    # Literature
    "LiteratureQueryPlan",
    "LiteratureQuery",
    "LiteratureQueryStatus",
]