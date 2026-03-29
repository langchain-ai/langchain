"""Google Search Console API tools.

Set ``GSC_MOCK=true`` in ``.env`` to return fixture data during development.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_MOCK_SEARCH_ANALYTICS = [
    {
        "keys": ["kitchen makers UK"],
        "clicks": 45,
        "impressions": 1200,
        "ctr": 0.0375,
        "position": 12.3,
    },
    {
        "keys": ["bespoke kitchens near me"],
        "clicks": 32,
        "impressions": 890,
        "ctr": 0.036,
        "position": 15.1,
    },
    {
        "keys": ["free room planner"],
        "clicks": 120,
        "impressions": 3400,
        "ctr": 0.035,
        "position": 8.2,
    },
    {
        "keys": ["kitchen layout planner"],
        "clicks": 85,
        "impressions": 2100,
        "ctr": 0.040,
        "position": 10.5,
    },
    {
        "keys": ["how much does a kitchen cost UK"],
        "clicks": 15,
        "impressions": 600,
        "ctr": 0.025,
        "position": 22.0,
    },
]

_MOCK_TOP_PAGES = [
    {
        "page": "https://www.kitchensdirectory.co.uk/",
        "clicks": 350,
        "impressions": 8500,
        "ctr": 0.041,
        "position": 14.2,
    },
    {
        "page": "https://www.kitchensdirectory.co.uk/kitchen-makers-london",
        "clicks": 120,
        "impressions": 3200,
        "ctr": 0.038,
        "position": 11.5,
    },
    {
        "page": "https://www.freeroomplanner.com/",
        "clicks": 580,
        "impressions": 15000,
        "ctr": 0.039,
        "position": 7.8,
    },
]


def _is_mock() -> bool:
    return os.getenv("GSC_MOCK", "false").lower() in ("true", "1", "yes")


def _build_service() -> Any:
    """Build an authenticated Google Search Console API service.

    Returns:
        A ``googleapiclient.discovery.Resource`` for the Search Console API.

    Raises:
        FileNotFoundError: If the service account JSON path is not set.
    """
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    sa_path = os.getenv("GSC_SERVICE_ACCOUNT_PATH", "")
    if not sa_path:
        msg = "GSC_SERVICE_ACCOUNT_PATH not set in environment"
        raise FileNotFoundError(msg)

    credentials = service_account.Credentials.from_service_account_file(
        sa_path,
        scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
    )
    return build("searchconsole", "v1", credentials=credentials)


def get_search_analytics(
    site_url: str,
    start_date: str,
    end_date: str,
    dimensions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Pull search analytics data from GSC.

    Args:
        site_url: The GSC property URL (e.g. ``https://www.kitchensdirectory.co.uk``).
        start_date: Start date in ``YYYY-MM-DD`` format.
        end_date: End date in ``YYYY-MM-DD`` format.
        dimensions: Dimensions to group by (e.g. ``["query"]``, ``["page"]``).

    Returns:
        List of row dicts with keys, clicks, impressions, ctr, position.
    """
    if dimensions is None:
        dimensions = ["query"]

    if _is_mock():
        return _MOCK_SEARCH_ANALYTICS

    service = _build_service()
    body: dict[str, Any] = {
        "startDate": start_date,
        "endDate": end_date,
        "dimensions": dimensions,
        "rowLimit": 1000,
    }
    response = (
        service.searchanalytics()
        .query(siteUrl=site_url, body=body)
        .execute()
    )
    return response.get("rows", [])


def get_top_queries(
    site_url: str, limit: int = 100
) -> list[dict[str, Any]]:
    """Get top search queries for a GSC property over the last 28 days.

    Args:
        site_url: The GSC property URL.
        limit: Maximum number of queries to return.

    Returns:
        List of query dicts sorted by clicks descending.
    """
    if _is_mock():
        return _MOCK_SEARCH_ANALYTICS[:limit]

    now = datetime.now(tz=timezone.utc)
    end = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    start = (now - timedelta(days=31)).strftime("%Y-%m-%d")

    rows = get_search_analytics(site_url, start, end, dimensions=["query"])
    rows.sort(key=lambda r: r.get("clicks", 0), reverse=True)
    return rows[:limit]


def get_top_pages(
    site_url: str, limit: int = 100
) -> list[dict[str, Any]]:
    """Get top pages for a GSC property over the last 28 days.

    Args:
        site_url: The GSC property URL.
        limit: Maximum number of pages to return.

    Returns:
        List of page dicts sorted by clicks descending.
    """
    if _is_mock():
        return _MOCK_TOP_PAGES[:limit]

    now = datetime.now(tz=timezone.utc)
    end = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    start = (now - timedelta(days=31)).strftime("%Y-%m-%d")

    rows = get_search_analytics(site_url, start, end, dimensions=["page"])
    rows.sort(key=lambda r: r.get("clicks", 0), reverse=True)
    return rows[:limit]
