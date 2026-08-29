"""MathModel AI — Phase 7A: Literature search providers.

BaseLiteratureSearchProvider abstraction + CrossrefSearchProvider.
Real search results only — no fabricated literature.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from mathmodel.literature import LiteratureRecord, normalize_doi

logger = logging.getLogger(__name__)


class BaseLiteratureSearchProvider(ABC):
    """Abstract interface for literature search."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search for literature. Returns list of raw result dicts."""
        ...

    @abstractmethod
    async def lookup_doi(self, doi: str) -> Optional[dict[str, Any]]:
        """Look up a DOI. Returns raw result dict or None."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Whether the provider is available."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    def normalize_result(self, raw: dict[str, Any]) -> LiteratureRecord:
        """Convert a raw search result to a LiteratureRecord."""
        return LiteratureRecord(
            title=raw.get("title", "") or "",
            authors=raw.get("authors", []),
            year=raw.get("year"),
            venue=raw.get("venue", ""),
            abstract=raw.get("abstract", ""),
            doi=raw.get("doi"),
            url=raw.get("url", ""),
            source=self.provider_name,
            publication_type=raw.get("type", "article"),
            retrieved_at=datetime.now(timezone.utc),
            is_fixture=False,
        )


class CrossrefSearchProvider(BaseLiteratureSearchProvider):
    """Crossref API search provider.

    Free, open REST API. No API key required for basic usage.
    https://api.crossref.org/
    """

    CROSSREF_BASE = "https://api.crossref.org"

    @property
    def provider_name(self) -> str:
        return "crossref"

    def health_check(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(
                f"{self.CROSSREF_BASE}/works?rows=0",
                timeout=10,
            )
            return True
        except Exception:
            return False

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search Crossref for works matching the query."""
        import json
        import urllib.request
        import urllib.parse

        params = urllib.parse.urlencode({
            "query": query,
            "rows": str(max_results),
        })
        url = f"{self.CROSSREF_BASE}/works?{params}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MathModelAI/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            logger.error("Crossref search failed: %s", e)
            return []

        results = []
        items = data.get("message", {}).get("items", [])
        for item in items:
            authors = []
            for author in item.get("author", [])[:5]:
                given = author.get("given", "")
                family = author.get("family", "")
                if given or family:
                    authors.append(f"{given} {family}".strip())

            results.append({
                "title": " ".join(item.get("title", [])),
                "authors": authors,
                "year": item.get("published-print", {}).get("date-parts", [[None]])[0][0],
                "venue": " ".join(item.get("container-title", [])),
                "abstract": item.get("abstract", "") or "",
                "doi": item.get("DOI"),
                "url": item.get("URL", ""),
                "type": item.get("type", "article"),
            })

        return results

    async def lookup_doi(self, doi: str) -> Optional[dict[str, Any]]:
        """Look up a specific DOI on Crossref."""
        import json
        import urllib.request

        norm = normalize_doi(doi)
        url = f"{self.CROSSREF_BASE}/works/{norm}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MathModelAI/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            logger.error("Crossref DOI lookup failed for %s: %s", doi, e)
            return None

        item = data.get("message", {})
        if not item:
            return None

        authors = []
        for author in item.get("author", [])[:5]:
            given = author.get("given", "")
            family = author.get("family", "")
            if given or family:
                authors.append(f"{given} {family}".strip())

        return {
            "title": " ".join(item.get("title", [])),
            "authors": authors,
            "year": item.get("published-print", {}).get("date-parts", [[None]])[0][0],
            "venue": " ".join(item.get("container-title", [])),
            "abstract": item.get("abstract", "") or "",
            "doi": item.get("DOI"),
            "url": item.get("URL", ""),
            "type": item.get("type", "article"),
        }