"""LangChain tools for the Decodo web scraping API.

This module provides two tools:

- ``DecodoWebScrapeTool``: Scrapes any URL and returns the page content as
  markdown (or raw HTML/text) using the ``universal`` target.
- ``DecodoSearchTool``: Searches Google, Amazon, or Reddit and returns
  structured JSON results.

Both tools call the Decodo HTTP API directly via ``httpx`` and authenticate
with a Basic token supplied via the ``decodo_api_token`` field or the
``DECODO_API_TOKEN`` environment variable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Type

import httpx
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr, model_validator

_DEFAULT_BASE_URL = "https://scraper-api.decodo.com"
_SCRAPE_PATH = "/v2/scrape"
_DEFAULT_TIMEOUT = 180.0  # seconds

_ENGINE_TARGET_MAP: dict[str, str] = {
    "google": "google_search",
    "amazon": "amazon_search",
    "reddit": "google_search",  # Reddit searches are done via Google
}

_REDDIT_SITE_FILTER = "site:reddit.com"


def _build_headers(token: str) -> dict[str, str]:
    """Build HTTP headers for a Decodo API request.

    Args:
        token: Raw Decodo API token string.

    Returns:
        Dictionary of HTTP headers.
    """
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-integration": "langchain",
    }


def _do_scrape(
    token: str,
    base_url: str,
    payload: dict[str, Any],
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST to the Decodo scrape endpoint and return the parsed JSON response.

    Args:
        token: Raw Decodo API token string.
        base_url: Base URL of the Decodo API.
        payload: Request body to send as JSON.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        RuntimeError: On timeout, network error, or non-2xx HTTP response.
    """
    url = f"{base_url}{_SCRAPE_PATH}"
    headers = _build_headers(token)

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"Decodo API request timed out after {timeout}s: {exc}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Decodo API network error: {exc}") from exc

    if not response.is_success:
        error_message: str
        try:
            body = response.json()
            error_message = body.get("message") or f"HTTP {response.status_code}"
        except Exception:
            error_message = f"HTTP {response.status_code}"
        raise RuntimeError(f"Decodo API error: {error_message}")

    return response.json()  # type: ignore[no-any-return]


def _extract_content(response: dict[str, Any]) -> str:
    """Pull the first result's content string from a Decodo API response.

    Args:
        response: Parsed JSON response from the Decodo API.

    Returns:
        Content string. Returns an empty string if no results are present.
    """
    results = response.get("results", [])
    if not results:
        return ""
    first = results[0]
    content = first.get("content", "")
    if isinstance(content, str):
        return content
    # Some targets return structured content as a dict/list; serialise it.
    return json.dumps(content, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _WebScrapeInput(BaseModel):
    url: str = Field(..., description="The full URL to scrape (must include scheme).")


class _SearchInput(BaseModel):
    query: str = Field(..., description="The search query string.")
    engine: str = Field(
        default="google",
        description=(
            "Search engine to use. Supported values: "
            "'google' (Google Search), 'amazon' (Amazon product search), "
            "'reddit' (Reddit posts via google_search). "
            "Defaults to 'google'."
        ),
    )
    num_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return (1-100).",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class DecodoWebScrapeTool(BaseTool):
    """Scrape any URL and return its content as markdown/text.

    The tool uses Decodo's ``universal`` target which handles JavaScript
    rendering, anti-bot protection, and proxy rotation automatically.

    Attributes:
        decodo_api_token: Decodo API token. Reads from the
            ``DECODO_API_TOKEN`` environment variable when not provided
            explicitly.
        base_url: Base URL of the Decodo Scraper API.

    Example::

        from langchain_decodo import DecodoWebScrapeTool

        tool = DecodoWebScrapeTool(decodo_api_token="YOUR_TOKEN")
        result = tool.run("https://example.com")
        print(result)
    """

    name: str = "decodo_scrape_url"
    description: str = (
        "Scrape the full content of any web page given its URL. "
        "Returns the page content as markdown or plain text. "
        "Handles JavaScript-rendered pages, CAPTCHAs, and geo-blocked content "
        "automatically. Use this when you need the complete text of a specific URL. "
        "Input: a valid URL string (must include http:// or https://)."
    )
    args_schema: Type[BaseModel] = _WebScrapeInput

    decodo_api_token: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Decodo API token. Reads from the ``DECODO_API_TOKEN`` environment "
            "variable when not provided explicitly."
        ),
    )
    base_url: str = Field(
        default=_DEFAULT_BASE_URL,
        description="Base URL of the Decodo Scraper API.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_api_token(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load ``DECODO_API_TOKEN`` from the environment if not explicitly set.

        Args:
            values: Raw field values before model construction.

        Returns:
            Updated field values with the token populated from env if needed.
        """
        if not values.get("decodo_api_token"):
            env_token = os.environ.get("DECODO_API_TOKEN", "")
            values["decodo_api_token"] = SecretStr(env_token)
        return values

    def _run(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Scrape the given URL and return its content.

        Args:
            url: The full URL to scrape (must include scheme).
            run_manager: Optional callback manager for tracing.

        Returns:
            Page content as a string (markdown or plain text).

        Raises:
            ValueError: If no API token is configured.
            RuntimeError: On API errors or network failures.
        """
        token = self.decodo_api_token.get_secret_value()
        if not token:
            raise ValueError(
                "Decodo API token is required. Set it via the ``decodo_api_token`` "
                "field or the ``DECODO_API_TOKEN`` environment variable."
            )

        payload: dict[str, Any] = {"target": "universal", "url": url}
        response = _do_scrape(token, self.base_url, payload)
        content = _extract_content(response)
        return content if content else "(No content returned by Decodo API)"


class DecodoSearchTool(BaseTool):
    """Search Google, Amazon, or Reddit and return structured JSON results.

    The tool maps the ``engine`` parameter to the appropriate Decodo target:

    * ``google``  → ``google_search`` target
    * ``amazon``  → ``amazon_search`` target
    * ``reddit``  → ``google_search`` with ``site:reddit.com`` prepended

    Results are returned as a JSON string containing a list of result objects.
    Each result object has at minimum a ``content`` field with the parsed data.

    Attributes:
        decodo_api_token: Decodo API token. Reads from the
            ``DECODO_API_TOKEN`` environment variable when not provided
            explicitly.
        engine: Default search engine (``google``, ``amazon``, or ``reddit``).

    Example::

        from langchain_decodo import DecodoSearchTool

        tool = DecodoSearchTool(decodo_api_token="YOUR_TOKEN")
        results = tool.run({"query": "best Python web scraping libraries"})
        print(results)
    """

    name: str = "decodo_search"
    description: str = (
        "Search the web using Google, Amazon, or Reddit and return structured results. "
        "Returns a JSON list of search results, each with 'content', 'url', and "
        "'status_code' fields. "
        "Input must be a JSON object with: "
        "'query' (required, the search string), "
        "'engine' (optional: 'google', 'amazon', or 'reddit'; default 'google'), "
        "'num_results' (optional: integer 1-100; default 10). "
        "Use 'amazon' to search for products, 'reddit' for community discussions."
    )
    args_schema: Type[BaseModel] = _SearchInput

    decodo_api_token: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Decodo API token. Reads from the ``DECODO_API_TOKEN`` environment "
            "variable when not provided explicitly."
        ),
    )
    engine: str = Field(
        default="google",
        description="Default search engine: 'google', 'amazon', or 'reddit'.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_api_token(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load ``DECODO_API_TOKEN`` from the environment if not explicitly set.

        Args:
            values: Raw field values before model construction.

        Returns:
            Updated field values with the token populated from env if needed.
        """
        if not values.get("decodo_api_token"):
            env_token = os.environ.get("DECODO_API_TOKEN", "")
            values["decodo_api_token"] = SecretStr(env_token)
        return values

    def _run(
        self,
        query: str,
        engine: str = "google",
        num_results: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute a search and return results as a JSON string.

        Args:
            query: The search query string.
            engine: Search engine to use (``google``, ``amazon``, ``reddit``).
            num_results: Maximum number of results to return (1-100).
            run_manager: Optional callback manager for tracing.

        Returns:
            JSON string — a list of result objects with ``content``,
            ``status_code``, and ``url`` fields.

        Raises:
            ValueError: If no API token is configured.
            RuntimeError: On API errors or network failures.
        """
        token = self.decodo_api_token.get_secret_value()
        if not token:
            raise ValueError(
                "Decodo API token is required. Set it via the ``decodo_api_token`` "
                "field or the ``DECODO_API_TOKEN`` environment variable."
            )

        engine = engine.lower().strip()
        target = _ENGINE_TARGET_MAP.get(engine, "google_search")

        # Prepend Reddit site filter when using the reddit pseudo-engine.
        effective_query = (
            f"{_REDDIT_SITE_FILTER} {query}" if engine == "reddit" else query
        )

        payload: dict[str, Any] = {
            "target": target,
            "query": effective_query,
            "limit": num_results,
        }

        response = _do_scrape(token, _DEFAULT_BASE_URL, payload)
        results = response.get("results", [])

        serialisable = []
        for entry in results:
            content = entry.get("content", "")
            serialisable.append(
                {
                    "content": content,
                    "status_code": entry.get("status_code"),
                    "url": entry.get("url", ""),
                }
            )

        return json.dumps(serialisable, ensure_ascii=False, indent=2)
