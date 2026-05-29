"""Shared utilities for the Bocha integration."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.utils import convert_to_secret_str

from langchain_bocha._client import DEFAULT_API_BASE, BochaClient


def initialize_client(values: dict[str, Any]) -> dict[str, Any]:
    """Initialize a `BochaClient` and inject it into *values*.

    This function is called inside ``model_validator(mode="before")`` by
    every class that needs a Bocha HTTP client (`ChatBocha`,
    `BochaSearchRun`, `BochaSearchResults`).

    It reads the API key from *values* or the ``BOCHA_API_KEY``
    environment variable and creates a `BochaClient` instance unless one
    has already been provided.

    Args:
        values: The dictionary of field values being validated.

    Returns:
        The updated *values* dictionary with ``client`` and
        ``bocha_api_key`` set.
    """
    bocha_api_key = (
        values.get("bocha_api_key")
        or values.get("api_key")
        or os.environ.get("BOCHA_API_KEY")
        or ""
    )
    values["bocha_api_key"] = convert_to_secret_str(bocha_api_key)

    api_key = (
        values["bocha_api_key"].get_secret_value() if values["bocha_api_key"] else ""
    )

    if not values.get("client"):
        base_url = values.get("base_url") or values.get("api_base", DEFAULT_API_BASE)
        timeout = values.get("timeout") or values.get("request_timeout") or 60
        max_retries = values.get("max_retries", 2)
        values["client"] = BochaClient(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout),
            max_retries=int(max_retries),
        )

    return values


def parse_search_results(
    response_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """Parse search results from a Bocha web-search API response.

    The actual Bocha API wraps the search payload under a ``data`` key.
    This function falls back to the top-level dict for backward
    compatibility.

    Args:
        response_json: The raw JSON response from the Bocha API.

    Returns:
        A list of dictionaries, each representing one search result.
    """
    data = response_json.get("data", response_json)
    web_pages = data.get("webPages", {})
    values = web_pages.get("value", [])

    return [
        {
            "id": val.get("id", ""),
            "title": val.get("name", ""),
            "link": val.get("url", ""),
            "display_url": val.get("displayUrl", ""),
            "snippet": val.get("snippet", ""),
            "summary": val.get("summary", ""),
            "site_name": val.get("siteName", ""),
            "site_icon": val.get("siteIcon", ""),
            "date_published": val.get("datePublished", ""),
            "date_last_crawled": val.get("dateLastCrawled", ""),
            "images": val.get("images", []),
            "language": val.get("language"),
            "is_family_friendly": val.get("isFamilyFriendly"),
            "is_navigational": val.get("isNavigational"),
        }
        for val in values
    ]


def format_search_results(results: list[dict[str, Any]]) -> str:
    """Format parsed search results into a human-readable string.

    Args:
        results: Parsed search result dictionaries.

    Returns:
        A formatted string representation.
    """
    if not results:
        return "No good Bocha Search Result Was Found"

    formatted = []
    for res in results:
        content = res["summary"] if res.get("summary") else res["snippet"]
        formatted.append(
            f"Title: {res['title']}\nLink: {res['link']}\nSnippet: {content}\n"
        )
    return "\n".join(formatted)
