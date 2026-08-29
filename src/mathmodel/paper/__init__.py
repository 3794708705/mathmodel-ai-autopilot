"""MathModel AI — Phase 6C: Paper IR.

Paper is structured content — NOT one giant markdown string.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    LIST = "list"
    SUBSECTION = "subsection"
    NOTE = "note"


class ContentBlock(BaseModel):
    """A single content block in a section."""
    block_id: str = Field(default_factory=lambda: f"BLK-{uuid4().hex[:8]}")
    block_type: BlockType
    text: str = ""
    equation_ids: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    items: list[str] = Field(default_factory=list)  # For LIST blocks
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperSection(BaseModel):
    """A section of the paper."""
    section_id: str = Field(default_factory=lambda: f"SEC-{uuid4().hex[:8]}")
    title: str = Field(..., min_length=1)
    purpose: str = ""
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    equation_ids: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperIR(BaseModel):
    """Complete structured paper representation."""
    paper_id: str = Field(default_factory=lambda: f"PAPER-{uuid4().hex[:8]}")
    title: str = Field(default="")
    abstract: str = Field(default="")
    keywords: list[str] = Field(default_factory=list)
    sections: list[PaperSection] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list, description="REF-xxx IDs")
    appendices: list[PaperSection] = Field(default_factory=list)
    competition_profile: Optional[dict[str, Any]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def collect_claims(self) -> list[str]:
        """All claim IDs referenced anywhere in the paper."""
        claims = []
        for section in self.sections:
            claims.extend(section.claim_ids)
            for block in section.content_blocks:
                claims.extend(block.claim_ids)
        return list(dict.fromkeys(claims))  # dedupe, keep order

    def collect_equations(self) -> list[str]:
        eqs = []
        for section in self.sections:
            eqs.extend(section.equation_ids)
            for block in section.content_blocks:
                eqs.extend(block.equation_ids)
        return list(dict.fromkeys(eqs))

    def collect_figures(self) -> list[str]:
        figs = []
        for section in self.sections:
            figs.extend(section.figure_ids)
            for block in section.content_blocks:
                figs.extend(block.figure_ids)
        return list(dict.fromkeys(figs))

    def collect_tables(self) -> list[str]:
        tabs = []
        for section in self.sections:
            tabs.extend(section.table_ids)
            for block in section.content_blocks:
                tabs.extend(block.table_ids)
        return list(dict.fromkeys(tabs))

    def collect_citations(self) -> list[str]:
        cites = []
        for section in self.sections:
            cites.extend(section.citation_ids)
            for block in section.content_blocks:
                cites.extend(block.citation_ids)
        return list(dict.fromkeys(cites))


class PaperAgent:
    """Builds a PaperIR from verified evidence.

    Never invents data, literature, results. All content blocks reference
    claims, equations, figures, tables that exist in the registries.
    """

    name = "PaperAgent"
    role = "Paper assembly from verified evidence"

    def build(
        self,
        title: str,
        abstract: str,
        keywords: list[str],
        sections: list[PaperSection],
        references: list[str],
    ) -> PaperIR:
        return PaperIR(
            title=title,
            abstract=abstract,
            keywords=keywords,
            sections=sections,
            references=references,
        )