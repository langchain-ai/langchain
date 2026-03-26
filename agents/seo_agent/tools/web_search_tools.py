"""Web search tools using Tavily for backlink prospecting and HARO monitoring.

Set ``TAVILY_MOCK=true`` for fixture data during development.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_MOCK_SEARCH_RESULTS = [
    {
        "title": "Best Kitchen Companies UK 2025 — Homebuilding & Renovating",
        "url": "https://homebuilding.co.uk/best-kitchen-companies-uk",
        "content": (
            "Our pick of the best kitchen companies in the UK for 2025, "
            "from bespoke to budget-friendly options."
        ),
        "score": 0.92,
    },
    {
        "title": "Useful Resources for Planning Your Kitchen — Real Homes",
        "url": "https://realhomes.com/kitchen-planning-resources",
        "content": (
            "A collection of recommended links and tools for anyone "
            "planning a kitchen renovation."
        ),
        "score": 0.88,
    },
    {
        "title": "Free Room Planner Tools Compared — TechRadar",
        "url": "https://techradar.com/best/free-room-planning-tools",
        "content": (
            "We tested the best free room planner tools available online, "
            "from RoomSketcher to Planner5D."
        ),
        "score": 0.85,
    },
]

_MOCK_UNLINKED_MENTIONS = [
    {
        "title": "How to Find a Kitchen Fitter — DIYnot Forum",
        "url": "https://diynot.com/threads/how-to-find-kitchen-fitter.12345/",
        "content": (
            "Someone mentioned kitchensdirectory.co.uk as a good place "
            "to start looking for kitchen companies."
        ),
        "score": 0.90,
        "has_link": False,
    },
]


def _is_mock() -> bool:
    # Default to mock only if TAVILY_API_KEY is not set
    explicit = os.getenv("TAVILY_MOCK")
    if explicit is not None:
        return explicit.lower() in ("true", "1", "yes")
    return not bool(os.getenv("TAVILY_API_KEY"))


def _get_tavily_client() -> Any:
    """Return a Tavily client instance.

    Returns:
        An initialised ``TavilyClient``.
    """
    from tavily import TavilyClient

    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def search(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Run a Tavily web search.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.

    Returns:
        List of result dicts with title, url, content, score.
    """
    if _is_mock():
        return _MOCK_SEARCH_RESULTS[:max_results]

    client = _get_tavily_client()
    response = client.search(query=query, max_results=max_results)
    return response.get("results", [])


def find_unlinked_mentions(
    domain: str, brand_terms: list[str] | None = None
) -> list[dict[str, Any]]:
    """Search for pages that mention a domain or brand without linking.

    Args:
        domain: The domain to search for mentions of.
        brand_terms: Additional brand terms to search (e.g. ``["kitchen directory UK"]``).

    Returns:
        List of result dicts with ``has_link`` field indicating linkage.
    """
    if _is_mock():
        return _MOCK_UNLINKED_MENTIONS

    queries = [f'"{domain}" -site:{domain}']
    if brand_terms:
        queries.extend(
            f'"{term}" -site:{domain}' for term in brand_terms
        )

    all_results: list[dict[str, Any]] = []
    client = _get_tavily_client()
    for q in queries:
        resp = client.search(query=q, max_results=10)
        for result in resp.get("results", []):
            result["has_link"] = False  # Would need scraping to verify
            all_results.append(result)

    # Deduplicate by URL
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in all_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)
    return unique


def search_resource_pages(niche: str = "kitchen") -> list[dict[str, Any]]:
    """Find resource/links pages relevant to a niche.

    Args:
        niche: The topic niche to search for resource pages in.

    Returns:
        List of resource page result dicts.
    """
    queries = [
        f'intitle:"useful resources" + "{niche}"',
        f'intitle:"recommended links" + "interior design"',
        f'"useful links" + "home renovation" + UK',
        f'inurl:resources + "{niche} design"',
    ]

    if _is_mock():
        return _MOCK_SEARCH_RESULTS[:2]

    client = _get_tavily_client()
    all_results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for q in queries:
        resp = client.search(query=q, max_results=5)
        for result in resp.get("results", []):
            if result["url"] not in seen:
                seen.add(result["url"])
                all_results.append(result)
    return all_results


def search_haro_requests(
    topics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search for journalist/HARO requests on relevant topics.

    Args:
        topics: Topics to filter for. Defaults to kitchen/home renovation topics.

    Returns:
        List of HARO request dicts.
    """
    if topics is None:
        topics = [
            "kitchen renovation",
            "interior design",
            "home improvement",
            "room planning",
            "kitchen cost",
        ]

    if _is_mock():
        return [
            {
                "title": "Looking for kitchen renovation expert quotes",
                "url": "https://helpareporter.com/example",
                "topic": "kitchen renovation",
                "deadline": "2026-04-01",
            },
        ]

    client = _get_tavily_client()
    results: list[dict[str, Any]] = []
    for topic in topics:
        resp = client.search(
            query=f"HARO journalist request {topic} UK",
            max_results=5,
        )
        for r in resp.get("results", []):
            r["topic"] = topic
            results.append(r)
    return results
