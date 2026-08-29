"""MathModel AI — Literature domain model.

Skeleton for literature search planning.
No real search tools yet — queries remain SEARCH_PENDING.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class LiteratureQueryStatus(str, Enum):
    """Status of a literature query."""
    SEARCH_PENDING = "SEARCH_PENDING"
    SEARCHING = "searching"
    COMPLETED = "completed"
    FAILED = "failed"


class LiteratureQuery(BaseModel):
    """A single literature search query.

    Describes what to search for and why, without fabricated results.
    """

    query_id: str = Field(default_factory=lambda: f"LQ-{uuid4().hex[:8]}")
    purpose: str = Field(..., min_length=1, description="Why this query is needed")
    keywords: list[str] = Field(default_factory=list)
    target_problem: Optional[str] = Field(default=None)
    target_method: Optional[str] = Field(default=None)
    target_application: Optional[str] = Field(default=None)
    candidate_ids: list[str] = Field(
        default_factory=list,
        description="Candidate models this query supports",
    )
    priority: int = Field(default=1, ge=1, description="1 = highest")
    status: LiteratureQueryStatus = LiteratureQueryStatus.SEARCH_PENDING
    metadata: dict[str, Any] = Field(default_factory=dict)


class LiteratureQueryPlan(BaseModel):
    """A plan for literature search, composed of multiple queries.

    This is a skeleton — no real search results are stored here.
    All queries are SEARCH_PENDING until a real search tool is available.
    """

    plan_id: str = Field(default_factory=lambda: f"LQP-{uuid4().hex[:8]}")
    queries: list[LiteratureQuery] = Field(default_factory=list)
    created_at: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def all_pending(self) -> bool:
        """Check if all queries are still pending."""
        return all(
            q.status == LiteratureQueryStatus.SEARCH_PENDING
            for q in self.queries
        )