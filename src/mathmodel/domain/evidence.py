"""MathModel AI — Evidence domain model.

Evidence items form the foundation of traceability.
Every fact, assumption, data source, and derivation is tracked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    """Classification of evidence items."""
    FACT = "FACT"
    DATA = "DATA"
    PROPOSED_ASSUMPTION = "PROPOSED_ASSUMPTION"
    ACCEPTED_ASSUMPTION = "ACCEPTED_ASSUMPTION"
    DERIVATION = "DERIVATION"


class EvidenceStatus(str, Enum):
    """Status of an evidence item."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"


class EvidenceItem(BaseModel):
    """A single piece of evidence in the reasoning chain.

    Facts cannot be silently downgraded to assumptions.
    Assumptions must be explicitly accepted before use.
    Derivations must reference their sources.
    """

    evidence_id: str = Field(default_factory=lambda: f"EVD-{uuid4().hex[:8]}")
    type: EvidenceType
    content: str = Field(..., min_length=1, description="The evidence statement")
    source: str = Field(..., description="Where this evidence came from")
    source_location: Optional[str] = Field(
        default=None, description="Specific location in source (page, section, etc.)"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in this evidence (0-1)"
    )
    status: EvidenceStatus = EvidenceStatus.ACTIVE
    created_by: str = Field(default="system", description="Agent or process that created this")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # For ACCEPTED_ASSUMPTION only
    accepted_by: Optional[str] = None
    accepted_at: Optional[datetime] = None
    acceptance_reason: Optional[str] = None

    # For DERIVATION only
    derived_from: list[str] = Field(
        default_factory=list,
        description="Evidence IDs this derivation references",
    )

    # Metadata
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_assumption(self) -> bool:
        return self.type in (EvidenceType.PROPOSED_ASSUMPTION, EvidenceType.ACCEPTED_ASSUMPTION)

    @property
    def is_accepted(self) -> bool:
        return self.type == EvidenceType.ACCEPTED_ASSUMPTION

    def accept(self, accepted_by: str, reason: str) -> "EvidenceItem":
        """Promote a PROPOSED_ASSUMPTION to ACCEPTED_ASSUMPTION."""
        if self.type != EvidenceType.PROPOSED_ASSUMPTION:
            raise ValueError(
                f"Cannot accept evidence of type {self.type.value}. "
                f"Only PROPOSED_ASSUMPTION can be accepted."
            )
        return self.model_copy(update={
            "type": EvidenceType.ACCEPTED_ASSUMPTION,
            "accepted_by": accepted_by,
            "accepted_at": datetime.now(timezone.utc),
            "acceptance_reason": reason,
        })

    def validate_invariants(self) -> list[str]:
        """Check evidence invariants. Returns list of violation messages."""
        violations = []

        if self.type == EvidenceType.ACCEPTED_ASSUMPTION:
            if not self.accepted_by:
                violations.append("ACCEPTED_ASSUMPTION must have accepted_by")
            if not self.acceptance_reason:
                violations.append("ACCEPTED_ASSUMPTION must have acceptance_reason")

        if self.type == EvidenceType.DERIVATION:
            if not self.derived_from:
                violations.append("DERIVATION must reference at least one source")

        return violations