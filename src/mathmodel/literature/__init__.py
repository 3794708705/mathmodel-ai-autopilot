"""MathModel AI — Phase 6D: Literature records and citation verification.

No fabricated literature. All references must come from real search
results or user-provided sources.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class CitationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONFLICT = "conflict"
    IDENTITY_MISMATCH = "identity_mismatch"


class LiteratureRecord(BaseModel):
    """A literature item from a real source."""
    literature_id: str = Field(default_factory=lambda: f"LIT-{uuid4().hex[:8]}")
    title: str = Field(..., min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    abstract: str = ""
    doi: Optional[str] = None
    url: str = ""
    source: str = Field(default="", description="Which provider/user supplied this")
    publication_type: str = "article"
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verification_status: CitationStatus = CitationStatus.UNVERIFIED
    is_fixture: bool = False  # True for test fixtures, never for real references
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Binds a claim to a literature record."""
    citation_id: str = Field(default_factory=lambda: f"CIT-{uuid4().hex[:8]}")
    literature_id: str
    claim_id: Optional[str] = None
    support_level: str = "supports"  # supports | contradicts | mentions
    excerpt: str = Field(default="", description="Brief supporting metadata/excerpt")
    status: CitationStatus = CitationStatus.UNVERIFIED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class LiteratureStore:
    def __init__(self):
        self._records: dict[str, LiteratureRecord] = {}
        self._citations: dict[str, Citation] = {}

    def add_record(self, record: LiteratureRecord) -> None:
        # Duplicate DOI detection
        if record.doi:
            for existing in self._records.values():
                if existing.doi == record.doi and existing.literature_id != record.literature_id:
                    raise ValueError(
                        f"Duplicate DOI {record.doi}: {existing.literature_id} vs {record.literature_id}"
                    )
        self._records[record.literature_id] = record

    def get_record(self, literature_id: str) -> Optional[LiteratureRecord]:
        return self._records.get(literature_id)

    def add_citation(self, citation: Citation) -> None:
        if citation.literature_id not in self._records:
            raise ValueError(
                f"Citation {citation.citation_id} references unknown literature "
                f"{citation.literature_id}"
            )
        self._citations[citation.citation_id] = citation

    def get_citation(self, citation_id: str) -> Optional[Citation]:
        return self._citations.get(citation_id)

    def all_records(self) -> list[LiteratureRecord]:
        return list(self._records.values())

    def all_citations(self) -> list[Citation]:
        return list(self._citations.values())


class CitationVerifier:
    """Verifies citations against literature records."""

    def __init__(self, store: LiteratureStore):
        self._store = store

    def verify(self, citation: Citation) -> list[str]:
        """Return list of issues (empty = verified)."""
        issues = []
        record = self._store.get_record(citation.literature_id)
        if record is None:
            return [f"Citation {citation.citation_id}: literature {citation.literature_id} not found"]

        if record.verification_status in (
            CitationStatus.IDENTITY_MISMATCH,
            CitationStatus.CONFLICT,
        ):
            issues.append(
                f"Citation {citation.citation_id}: literature has {record.verification_status.value}"
            )
        if record.is_fixture:
            issues.append(
                f"Citation {citation.citation_id}: literature is a TEST_FIXTURE, not a real reference"
            )
        return issues

    def verify_all(self) -> list[str]:
        issues = []
        for citation in self._store.all_citations():
            issues.extend(self.verify(citation))
        return issues

    def verifiable_bibliography(self) -> list[LiteratureRecord]:
        """Records that may enter the final bibliography."""
        result = []
        for record in self._store.all_records():
            if record.verification_status == CitationStatus.VERIFIED and not record.is_fixture:
                result.append(record)
            elif record.verification_status == CitationStatus.PARTIALLY_VERIFIED and not record.is_fixture:
                result.append(record)
        return result