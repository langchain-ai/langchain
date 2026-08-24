"""LangChain document loader for the Decodo web scraping API.

``DecodoLoader`` accepts one or more URLs and uses the Decodo ``universal``
target to fetch their content. Each URL is yielded as a
``langchain_core.documents.Document`` with:

* ``page_content`` — the scraped text/markdown of the page.
* ``metadata``     — ``{"source": ..., "url": ..., "status_code": ...}``.

Typical usage::

    from langchain_decodo import DecodoLoader

    loader = DecodoLoader(
        urls=["https://example.com", "https://news.ycombinator.com"],
        api_token="YOUR_TOKEN",
    )
    docs = loader.load()
    for doc in docs:
        print(doc.metadata["url"], len(doc.page_content))
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import httpx
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pydantic import SecretStr

_API_BASE = "https://scraper-api.decodo.com"
_SCRAPE_PATH = "/v2/scrape"
_DEFAULT_TIMEOUT = 180.0


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


def _scrape_url(
    token: str,
    url: str,
    timeout: float,
) -> dict[str, Any]:
    """Scrape a single URL and return the raw API response dict.

    Args:
        token: Raw Decodo API token string.
        url: Full URL to scrape (must include scheme).
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response from the Decodo API.

    Raises:
        RuntimeError: On timeout, network error, or non-2xx HTTP response.
    """
    endpoint = f"{_API_BASE}{_SCRAPE_PATH}"
    payload: dict[str, Any] = {"target": "universal", "url": url}

    try:
        response = httpx.post(
            endpoint,
            headers=_build_headers(token),
            json=payload,
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"Decodo API request for '{url}' timed out after {timeout}s"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Decodo API network error while fetching '{url}': {exc}"
        ) from exc

    if not response.is_success:
        try:
            body = response.json()
            msg = body.get("message") or f"HTTP {response.status_code}"
        except Exception:
            msg = f"HTTP {response.status_code}"
        raise RuntimeError(f"Decodo API error for '{url}': {msg}")

    return response.json()  # type: ignore[no-any-return]


def _result_to_document(url: str, result: dict[str, Any]) -> Document:
    """Convert a single Decodo result entry dict into a LangChain Document.

    Args:
        url: The source URL that was scraped.
        result: A single entry from the ``results`` list in the API response.

    Returns:
        A ``Document`` with ``page_content`` and populated ``metadata``.
    """
    raw_content = result.get("content", "")
    if isinstance(raw_content, str):
        page_content = raw_content
    else:
        # Structured content (parsed JSON) — serialise so it is always a str.
        page_content = json.dumps(raw_content, ensure_ascii=False, indent=2)

    metadata: dict[str, Any] = {
        "source": url,
        "url": url,
        "status_code": result.get("status_code"),
    }
    # Carry through any extra result-level fields that may be useful downstream.
    for field in ("task_id", "created_at", "updated_at"):
        if field in result:
            metadata[field] = result[field]

    return Document(page_content=page_content, metadata=metadata)


class DecodoLoader(BaseLoader):
    """Load web pages as LangChain Documents using the Decodo scraping API.

    Args:
        urls: A single URL string or a list of URL strings to scrape.
        api_token: Decodo API token. Falls back to the ``DECODO_API_TOKEN``
            environment variable when omitted.
        timeout: HTTP request timeout in seconds (default 180).
        continue_on_error: When ``True`` (default), skip URLs that fail and
            continue loading the remaining ones. When ``False``, raise the
            first exception encountered.

    Example::

        loader = DecodoLoader(
            urls="https://example.com",
            api_token="YOUR_TOKEN",
        )
        docs = loader.load()
    """

    def __init__(
        self,
        urls: list[str] | str,
        api_token: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        continue_on_error: bool = True,
    ) -> None:
        if isinstance(urls, str):
            urls = [urls]
        self._urls: list[str] = urls

        raw_token = api_token or os.environ.get("DECODO_API_TOKEN", "")
        if not raw_token:
            raise ValueError(
                "Decodo API token is required. Provide it via the ``api_token`` "
                "argument or the ``DECODO_API_TOKEN`` environment variable."
            )
        self._token: SecretStr = SecretStr(raw_token)
        self._timeout = timeout
        self._continue_on_error = continue_on_error

    def lazy_load(self) -> Iterator[Document]:
        """Yield Documents one at a time, fetching each URL sequentially.

        Yields:
            One ``Document`` per scraped result. If ``continue_on_error`` is
            ``True``, failed URLs yield an empty Document with ``error`` in
            metadata instead of raising.
        """
        token = self._token.get_secret_value()

        for url in self._urls:
            try:
                api_response = _scrape_url(
                    token=token,
                    url=url,
                    timeout=self._timeout,
                )
            except RuntimeError as exc:
                if self._continue_on_error:
                    yield Document(
                        page_content="",
                        metadata={
                            "source": url,
                            "url": url,
                            "status_code": None,
                            "error": str(exc),
                        },
                    )
                    continue
                raise

            results: list[dict[str, Any]] = api_response.get("results", [])
            if not results:
                yield Document(
                    page_content="",
                    metadata={"source": url, "url": url, "status_code": None},
                )
                continue

            for result in results:
                yield _result_to_document(url, result)
