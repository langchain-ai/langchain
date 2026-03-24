"""Ahrefs API client and LangChain tool wrappers.

Set ``AHREFS_MOCK=true`` in ``.env`` to use fixture data during development.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock / fixture data
# ---------------------------------------------------------------------------

_MOCK_KEYWORDS = [
    {
        "keyword": "bespoke kitchen manufacturers UK",
        "volume": 720,
        "kd": 28,
        "cpc": 2.10,
        "intent": "commercial",
        "country": "gb",
    },
    {
        "keyword": "kitchen makers near me",
        "volume": 1300,
        "kd": 35,
        "cpc": 1.80,
        "intent": "transactional",
        "country": "gb",
    },
    {
        "keyword": "handmade kitchens UK",
        "volume": 880,
        "kd": 22,
        "cpc": 1.50,
        "intent": "commercial",
        "country": "gb",
    },
    {
        "keyword": "how to plan a kitchen layout",
        "volume": 2400,
        "kd": 18,
        "cpc": 0.90,
        "intent": "informational",
        "country": "gb",
    },
    {
        "keyword": "kitchen cost calculator UK",
        "volume": 590,
        "kd": 15,
        "cpc": 3.20,
        "intent": "transactional",
        "country": "gb",
    },
]

_MOCK_COMPETITORS = [
    {"domain": "houzz.com", "common_keywords": 245, "traffic": 150000},
    {"domain": "checkatrade.com", "common_keywords": 180, "traffic": 320000},
    {"domain": "trustatrader.com", "common_keywords": 120, "traffic": 95000},
]

_MOCK_CONTENT_GAP = [
    {
        "keyword": "best kitchen companies Manchester",
        "volume": 480,
        "kd": 25,
        "competitors_ranking": ["houzz.com", "checkatrade.com"],
        "top_url": "https://houzz.com/uk/kitchen-companies-manchester",
    },
    {
        "keyword": "kitchen renovation costs 2025",
        "volume": 1200,
        "kd": 30,
        "competitors_ranking": ["checkatrade.com", "rated.co.uk"],
        "top_url": "https://checkatrade.com/blog/kitchen-costs",
    },
]

_MOCK_BACKLINKS = [
    {
        "referring_domain": "homebuilding.co.uk",
        "page_url": "https://homebuilding.co.uk/best-kitchen-companies",
        "anchor": "kitchen directory",
        "dr": 72,
        "traffic": 8500,
        "dofollow": True,
    },
    {
        "referring_domain": "realhomes.com",
        "page_url": "https://realhomes.com/advice/kitchen-planning-guide",
        "anchor": "plan your kitchen",
        "dr": 68,
        "traffic": 12000,
        "dofollow": True,
    },
]

_MOCK_BROKEN_BACKLINKS = [
    {
        "referring_page": "https://idealhome.co.uk/kitchen/resources",
        "dead_url": "https://oldcompetitor.com/kitchen-guide",
        "anchor": "kitchen planning guide",
        "dr": 65,
        "traffic": 5000,
    },
]

_MOCK_RANK_TRACKING = [
    {
        "keyword": "kitchen makers UK",
        "position": 12,
        "previous_position": 15,
        "url": "https://www.kitchensdirectory.co.uk/",
        "volume": 1300,
    },
    {
        "keyword": "free room planner",
        "position": 8,
        "previous_position": 11,
        "url": "https://www.freeroomplanner.com/",
        "volume": 2400,
    },
]

_MOCK_CONTENT_EXPLORER = [
    {
        "title": "Best Kitchen Companies in the UK — 2025 Roundup",
        "url": "https://homebuilding.co.uk/best-kitchen-companies-uk",
        "dr": 72,
        "traffic": 4500,
        "word_count": 2800,
        "published": "2025-09-15",
    },
    {
        "title": "Free Room Planning Tools Compared",
        "url": "https://techradar.com/best/room-planning-tools",
        "dr": 85,
        "traffic": 18000,
        "word_count": 3200,
        "published": "2025-11-02",
    },
]


def _is_mock() -> bool:
    return os.getenv("AHREFS_MOCK", "false").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# HTTP client with retries
# ---------------------------------------------------------------------------


class AhrefsClient:
    """Thin wrapper around the Ahrefs REST API with retry logic."""

    BASE_URL = "https://api.ahrefs.com/v4"

    def __init__(self) -> None:
        self.api_key = os.getenv("AHREFS_API_KEY", "")
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict:
        """Send a GET request with retry and rate-limit handling.

        Args:
            endpoint: API path (e.g. ``/keywords-explorer/overview``).
            params: Query parameters.

        Returns:
            Parsed JSON response dict.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors.
        """
        resp = self._client.get(endpoint, params=params or {})
        if resp.status_code == 429:
            logger.warning("Ahrefs rate limit hit, retrying...")
            raise httpx.HTTPStatusError(
                "Rate limited",
                request=resp.request,
                response=resp,
            )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()


_ahrefs: AhrefsClient | None = None


def _get_client() -> AhrefsClient:
    global _ahrefs  # noqa: PLW0603
    if _ahrefs is None:
        _ahrefs = AhrefsClient()
    return _ahrefs


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


@tool
def get_keyword_overview(
    keywords: list[str], country: str = "gb"
) -> list[dict[str, Any]]:
    """Get Ahrefs keyword overview metrics for a list of keywords.

    Args:
        keywords: List of keyword strings to look up.
        country: Two-letter country code (default ``gb``).

    Returns:
        List of dicts with keyword, volume, kd, cpc, intent, country.
    """
    if _is_mock():
        return [kw for kw in _MOCK_KEYWORDS if kw["keyword"] in keywords] or [
            {
                "keyword": k,
                "volume": 500,
                "kd": 25,
                "cpc": 1.50,
                "intent": "commercial",
                "country": country,
            }
            for k in keywords
        ]
    client = _get_client()
    results = []
    for kw in keywords:
        data = client.get(
            "/keywords-explorer/overview",
            {"keyword": kw, "country": country},
        )
        results.append(data)
    return results


@tool
def get_keyword_ideas(
    seed_keyword: str, country: str = "gb"
) -> list[dict[str, Any]]:
    """Get related keyword ideas from Ahrefs for a seed keyword.

    Args:
        seed_keyword: The starting keyword to find ideas for.
        country: Two-letter country code (default ``gb``).

    Returns:
        List of keyword opportunity dicts.
    """
    if _is_mock():
        return [
            kw
            for kw in _MOCK_KEYWORDS
            if any(
                word in kw["keyword"]
                for word in seed_keyword.lower().split()
            )
        ] or _MOCK_KEYWORDS
    client = _get_client()
    return client.get(
        "/keywords-explorer/related-terms",
        {"keyword": seed_keyword, "country": country},
    ).get("keywords", [])


@tool
def get_competing_domains(target: str) -> list[dict[str, Any]]:
    """Get top organic competitors for a domain from Ahrefs.

    Args:
        target: The target domain (e.g. ``kitchensdirectory.co.uk``).

    Returns:
        List of competitor dicts with domain, common_keywords, traffic.
    """
    if _is_mock():
        return _MOCK_COMPETITORS
    client = _get_client()
    return client.get(
        "/site-explorer/competing-domains",
        {"target": target},
    ).get("domains", [])


@tool
def get_content_gap(
    target: str, competitors: list[str]
) -> list[dict[str, Any]]:
    """Run a content gap analysis between target and competitors.

    Args:
        target: Your domain.
        competitors: List of competitor domains.

    Returns:
        List of gap keyword dicts.
    """
    if _is_mock():
        return _MOCK_CONTENT_GAP
    client = _get_client()
    return client.get(
        "/site-explorer/content-gap",
        {"target": target, "competitors": ",".join(competitors)},
    ).get("keywords", [])


@tool
def get_backlinks(
    target: str, dr_min: int = 30, mode: str = "dofollow"
) -> list[dict[str, Any]]:
    """Get backlinks for a domain from Ahrefs.

    Args:
        target: The domain to check backlinks for.
        dr_min: Minimum domain rating filter.
        mode: Link type — ``dofollow`` or ``all``.

    Returns:
        List of backlink dicts.
    """
    if _is_mock():
        return [bl for bl in _MOCK_BACKLINKS if bl["dr"] >= dr_min]
    client = _get_client()
    return client.get(
        "/site-explorer/backlinks",
        {"target": target, "dr_min": dr_min, "mode": mode},
    ).get("backlinks", [])


@tool
def get_broken_backlinks(target: str) -> list[dict[str, Any]]:
    """Get broken backlinks pointing to a competitor domain.

    Args:
        target: The competitor domain to check for broken links.

    Returns:
        List of broken backlink opportunity dicts.
    """
    if _is_mock():
        return _MOCK_BROKEN_BACKLINKS
    client = _get_client()
    return client.get(
        "/site-explorer/broken-backlinks",
        {"target": target},
    ).get("backlinks", [])


@tool
def get_rank_tracking(target: str) -> list[dict[str, Any]]:
    """Get rank tracking data for a domain from Ahrefs.

    Args:
        target: The domain to track rankings for.

    Returns:
        List of rank tracking dicts with keyword, position, etc.
    """
    if _is_mock():
        return _MOCK_RANK_TRACKING
    client = _get_client()
    return client.get(
        "/rank-tracker/keywords",
        {"target": target},
    ).get("keywords", [])


@tool
def search_content_explorer(query: str) -> list[dict[str, Any]]:
    """Search Ahrefs Content Explorer for pages matching a query.

    Args:
        query: The search query string.

    Returns:
        List of content result dicts with title, url, dr, traffic, etc.
    """
    if _is_mock():
        return _MOCK_CONTENT_EXPLORER
    client = _get_client()
    return client.get(
        "/content-explorer/search",
        {"query": query},
    ).get("results", [])
