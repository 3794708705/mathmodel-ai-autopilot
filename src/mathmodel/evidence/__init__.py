"""MathModel AI — Phase 6A: Evidence and Claim system.

Establishes Claim → Evidence → Source relationships.
Core principle: NO EVIDENCE → NO CLAIM.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class EvidenceSourceType(str, Enum):
    PROBLEM_FACT = "problem_fact"
    DATA = "data"
    ASSUMPTION = "assumption"
    DERIVATION = "derivation"
    MATHEMATICAL_MODEL = "mathematical_model"
    EQUATION = "equation"
    PARAMETER = "parameter"
    EXECUTION = "execution"
    SOLVER_RESULT = "solver_result"
    VALIDATION = "validation"
    SENSITIVITY = "sensitivity"
    ROBUSTNESS = "robustness"
    RED_TEAM = "red_team"
    LITERATURE = "literature"
    FIGURE = "figure"
    TABLE = "table"


class ClaimType(str, Enum):
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    MATHEMATICAL = "mathematical"
    COMPARATIVE = "comparative"
    INTERPRETIVE = "interpretive"
    LITERATURE_SUPPORTED = "literature_supported"
    CONCLUSION = "conclusion"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class EvidenceRef(BaseModel):
    """Reference to an evidence source."""
    evidence_id: str = Field(default_factory=lambda: f"EVD-{uuid4().hex[:8]}")
    source_type: EvidenceSourceType
    source_id: str = Field(..., min_length=1, description="ID of the source object")
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """A claim made in the paper, tied to evidence."""

    claim_id: str = Field(default_factory=lambda: f"CLAIM-{uuid4().hex[:8]}")
    text: str = Field(..., min_length=1)
    claim_type: ClaimType = ClaimType.FACTUAL
    importance: str = "normal"  # critical | high | normal | low
    evidence_ids: list[str] = Field(default_factory=list)
    verification_status: ClaimStatus = ClaimStatus.UNVERIFIED
    section_id: Optional[str] = None
    created_by: str = Field(default="system")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_importance(self) -> "Claim":
        """Important claims must have evidence."""
        # Validation deferred to EvidenceStore (needs registry context)
        return self


class EvidenceStore:
    """Stores and validates evidence and claims."""

    # Required evidence types per claim type (for important claims).
    # e.g. a NUMERICAL claim must cite SOLVER_RESULT or EXECUTION,
    # not just a PROBLEM_FACT.
    CLAIM_TYPE_EVIDENCE_REQUIREMENTS: dict[ClaimType, list[EvidenceSourceType]] = {
        ClaimType.NUMERICAL: [EvidenceSourceType.SOLVER_RESULT, EvidenceSourceType.EXECUTION],
        ClaimType.COMPARATIVE: [EvidenceSourceType.SOLVER_RESULT, EvidenceSourceType.EXECUTION],
        ClaimType.MATHEMATICAL: [EvidenceSourceType.EQUATION, EvidenceSourceType.MATHEMATICAL_MODEL, EvidenceSourceType.DERIVATION],
        ClaimType.FACTUAL: [EvidenceSourceType.PROBLEM_FACT, EvidenceSourceType.DATA],
        ClaimType.INTERPRETIVE: [EvidenceSourceType.VALIDATION, EvidenceSourceType.SENSITIVITY, EvidenceSourceType.ROBUSTNESS, EvidenceSourceType.RED_TEAM],
        ClaimType.LITERATURE_SUPPORTED: [EvidenceSourceType.LITERATURE],
        ClaimType.CONCLUSION: [EvidenceSourceType.VALIDATION, EvidenceSourceType.SOLVER_RESULT],
    }

    def __init__(self):
        self._evidence: dict[str, EvidenceRef] = {}
        self._claims: dict[str, Claim] = {}

    # ── Evidence management ──────────────────────────────────

    def register_evidence(self, ref: EvidenceRef) -> None:
        if ref.evidence_id in self._evidence:
            raise ValueError(f"Duplicate evidence ID: {ref.evidence_id}")
        self._evidence[ref.evidence_id] = ref

    def get_evidence(self, evidence_id: str) -> Optional[EvidenceRef]:
        return self._evidence.get(evidence_id)

    def has_evidence(self, evidence_id: str) -> bool:
        return evidence_id in self._evidence

    def list_evidence(self) -> list[EvidenceRef]:
        return list(self._evidence.values())

    def delete_evidence(self, evidence_id: str) -> None:
        """Delete evidence and invalidate all claims referencing it."""
        if evidence_id not in self._evidence:
            raise KeyError(f"Unknown evidence ID: {evidence_id}")
        del self._evidence[evidence_id]
        # Invalidate dependent claims
        for claim in self._claims.values():
            if evidence_id in claim.evidence_ids:
                claim.verification_status = ClaimStatus.UNVERIFIED

    # ── Claim management ─────────────────────────────────────

    def register_claim(self, claim: Claim) -> None:
        if claim.claim_id in self._claims:
            raise ValueError(f"Duplicate claim ID: {claim.claim_id}")

        # Validate evidence references
        missing = [eid for eid in claim.evidence_ids if not self.has_evidence(eid)]
        if missing:
            raise ValueError(
                f"Claim {claim.claim_id} references unknown evidence: {missing}"
            )

        # Gate: important claims without evidence → UNSUPPORTED
        if claim.importance in ("critical", "high") and not claim.evidence_ids:
            claim.verification_status = ClaimStatus.UNSUPPORTED

        self._claims[claim.claim_id] = claim

    def check_evidence_type_match(self, claim: Claim) -> bool:
        """Whether the claim's evidence types match its claim type.

        A NUMERICAL claim backed only by PROBLEM_FACT is not sufficient.
        """
        required = self.CLAIM_TYPE_EVIDENCE_REQUIREMENTS.get(claim.claim_type)
        if not required:
            return True
        cited_types = {
            self._evidence[eid].source_type
            for eid in claim.evidence_ids
            if eid in self._evidence
        }
        return bool(cited_types & set(required))

    def claims_with_invalid_evidence_type(self) -> list[Claim]:
        """Important claims whose evidence types do not match claim type."""
        result = []
        for claim in self._claims.values():
            if claim.importance in ("critical", "high") and claim.evidence_ids:
                if not self.check_evidence_type_match(claim):
                    result.append(claim)
        return result

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self._claims.get(claim_id)

    def list_claims(self) -> list[Claim]:
        return list(self._claims.values())

    def unsupported_claims(self) -> list[Claim]:
        """All claims that are UNSUPPORTED."""
        return [c for c in self._claims.values()
                if c.verification_status == ClaimStatus.UNSUPPORTED]

    def claims_by_section(self, section_id: str) -> list[Claim]:
        return [c for c in self._claims.values() if c.section_id == section_id]


class ClaimGate:
    """Blocks unsupported claims from entering the paper."""

    @staticmethod
    def check(claims: list[Claim]) -> list[str]:
        """Return blocking issues for unsupported important claims."""
        issues = []
        for claim in claims:
            if claim.verification_status == ClaimStatus.UNSUPPORTED:
                issues.append(
                    f"Claim {claim.claim_id} is UNSUPPORTED (importance={claim.importance}): "
                    f"{claim.text[:80]}"
                )
            elif claim.verification_status == ClaimStatus.UNVERIFIED and \
                    claim.importance == "critical":
                issues.append(
                    f"Claim {claim.claim_id} is UNVERIFIED but critical: {claim.text[:80]}"
                )
        return issues

    @staticmethod
    def can_enter_paper(claim: Claim) -> bool:
        """Whether a claim may enter the final paper."""
        if claim.importance in ("critical", "high"):
            return claim.verification_status == ClaimStatus.SUPPORTED
        return claim.verification_status in (
            ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED
        )