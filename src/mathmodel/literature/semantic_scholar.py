"""MathModel AI — Semantic Scholar literature search provider.

Real academic search with abstracts. Uses the Semantic Scholar
REST API (free tier, no key required for basic search).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

from mathmodel.literature.search import BaseLiteratureSearchProvider
from mathmodel.literature import LiteratureRecord

logger = logging.getLogger(__name__)


class SemanticScholarProvider(BaseLiteratureSearchProvider):
    """Real literature search via Semantic Scholar API.

    Returns records with abstracts when available.
    Free tier: no API key required for basic search (rate-limited).
    """

    name = "Semantic Scholar"
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    @property
    def provider_name(self) -> str:
        return "semantic_scholar"

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        import aiohttp
        try:
            url = f"{self.BASE_URL}/paper/search"
            params = {
                "query": query,
                "limit": min(max_results, 10),
                "fields": "title,authors,year,venue,abstract,externalIds,url",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status != 200:
                        logger.warning("Semantic Scholar returned %d", resp.status)
                        return []
                    data = await resp.json()
                    return data.get("data", [])
        except Exception as e:
            logger.warning("Semantic Scholar search failed: %s", e)
            return []

    async def lookup_doi(self, doi: str) -> Optional[dict]:
        import aiohttp
        try:
            clean_doi = doi.replace("https://doi.org/", "").strip()
            url = f"{self.BASE_URL}/paper/DOI:{quote(clean_doi, safe='')}"
            params = {"fields": "title,authors,year,venue,abstract,externalIds,url"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.json()
        except Exception as e:
            logger.warning("Semantic Scholar DOI lookup failed: %s", e)
            return None

    async def normalize_result(self, raw: dict) -> LiteratureRecord:
        external_ids = raw.get("externalIds", {}) or {}
        authors_list = raw.get("authors", []) or []
        authors = [a.get("name", "") for a in authors_list]

        return LiteratureRecord(
            literature_id=f"SS-{raw.get('paperId', 'unknown')[:12]}",
            title=raw.get("title", ""),
            authors=authors,
            year=raw.get("year"),
            venue=raw.get("venue", ""),
            abstract=raw.get("abstract", ""),
            doi=external_ids.get("DOI", ""),
            url=raw.get("url", ""),
            retrieval_source="semantic_scholar",
            retrieved_at=datetime.now(timezone.utc),
            is_fixture=False,
        )

    async def health_check(self) -> bool:
        try:
            results = await self.search("test", max_results=1)
            return True
        except Exception:
            return False