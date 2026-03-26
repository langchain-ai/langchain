"""CRW document loader for LangChain."""

from __future__ import annotations

import os
import time
from typing import Any, Iterator, Literal, Optional

import requests
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class CrwLoader(BaseLoader):
    """Load documents using CRW web scraper.

    CRW is a high-performance, Firecrawl-compatible web scraper.
    Self-hosted (crw-server) or cloud (fastcrw.com).

    Setup:
        Install ``langchain-crw`` and start crw-server locally or get an
        API key from https://fastcrw.com.

        .. code-block:: bash

            pip install langchain-crw
            cargo install crw-server && crw-server

    Instantiate:
        .. code-block:: python

            from langchain_crw import CrwLoader

            # Self-hosted (no API key needed)
            loader = CrwLoader(url="https://example.com", mode="scrape")

            # Cloud (fastcrw.com)
            loader = CrwLoader(
                url="https://example.com",
                api_key="fc-...",
                api_url="https://fastcrw.com/api",
                mode="crawl",
            )

    Lazy load:
        .. code-block:: python

            for doc in loader.lazy_load():
                print(doc.page_content[:100])
                print(doc.metadata)
    """

    def __init__(
        self,
        url: str,
        *,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        mode: Literal["scrape", "crawl", "map"] = "scrape",
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize CrwLoader.

        Args:
            url: The URL to scrape, crawl, or map.
            api_key: Bearer token for authentication.
                Read from ``CRW_API_KEY`` env var if not provided.
                Not required for self-hosted without auth.
            api_url: Base URL of CRW server.
                Read from ``CRW_API_URL`` env var if not provided.
                Defaults to ``http://localhost:3000``.
            mode: Operation mode - ``"scrape"``, ``"crawl"``, or ``"map"``.
            params: Additional parameters passed to the CRW API.
        """
        self.url = url
        self.api_key = api_key or os.getenv("CRW_API_KEY")
        self.api_url = (
            api_url or os.getenv("CRW_API_URL") or "http://localhost:3000"
        ).rstrip("/")
        self.mode = mode
        self.params = params or {}
        self._session = requests.Session()
        if self.api_key:
            self._session.headers["Authorization"] = f"Bearer {self.api_key}"
        self._session.headers["Content-Type"] = "application/json"

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents from CRW."""
        if self.mode == "scrape":
            return self._scrape()
        elif self.mode == "crawl":
            return self._crawl()
        elif self.mode == "map":
            return self._map()
        else:
            msg = (
                f"Invalid mode '{self.mode}'. "
                "Must be 'scrape', 'crawl', or 'map'."
            )
            raise ValueError(msg)

    def _scrape(self) -> Iterator[Document]:
        """Scrape a single URL."""
        body: dict[str, Any] = {"url": self.url}
        body.update(self._build_api_params())
        response = self._request("POST", "/v1/scrape", json=body)

        data = response.get("data", {})
        if not data:
            return

        doc = self._parse_document(data)
        if doc.page_content:
            yield doc

    def _crawl(self) -> Iterator[Document]:
        """Crawl a site via async job polling."""
        body: dict[str, Any] = {"url": self.url}
        body.update(self._build_api_params())

        # Start crawl job
        start_response = self._request("POST", "/v1/crawl", json=body)
        job_id = start_response.get("id")
        if not job_id:
            msg = (
                "CRW crawl did not return a job ID. "
                f"Response: {start_response}"
            )
            raise ValueError(msg)

        # Poll for completion
        poll_interval = self.params.get("poll_interval", 2)
        timeout = self.params.get("timeout", 300)
        elapsed = 0.0

        while elapsed < timeout:
            status_response = self._request("GET", f"/v1/crawl/{job_id}")
            status = status_response.get("status")

            if status == "completed":
                for page in status_response.get("data", []):
                    doc = self._parse_document(page)
                    if doc.page_content:
                        yield doc
                return

            if status == "failed":
                msg = (
                    f"CRW crawl job '{job_id}' failed. "
                    f"Response: {status_response}"
                )
                raise RuntimeError(msg)

            time.sleep(poll_interval)
            elapsed += poll_interval

        msg = f"CRW crawl job '{job_id}' timed out after {timeout}s."
        raise TimeoutError(msg)

    def _map(self) -> Iterator[Document]:
        """Discover URLs on a site."""
        body: dict[str, Any] = {"url": self.url}
        body.update(self._build_api_params())
        response = self._request("POST", "/v1/map", json=body)

        for link in response.get("links", []):
            if link:
                yield Document(page_content=link, metadata={})

    def _build_api_params(self) -> dict[str, Any]:
        """Convert snake_case params to camelCase API fields."""
        mapping = {
            "formats": "formats",
            "only_main_content": "onlyMainContent",
            "render_js": "renderJs",
            "wait_for": "waitFor",
            "include_tags": "includeTags",
            "exclude_tags": "excludeTags",
            "headers": "headers",
            "css_selector": "cssSelector",
            "xpath": "xpath",
            "json_schema": "jsonSchema",
            "proxy": "proxy",
            "stealth": "stealth",
            "max_depth": "maxDepth",
            "max_pages": "maxPages",
            "use_sitemap": "useSitemap",
        }
        result: dict[str, Any] = {}
        for key, value in self.params.items():
            api_key = mapping.get(key, key)
            # Skip loader-internal params that aren't API fields
            if key in ("poll_interval", "timeout"):
                continue
            result[api_key] = value
        return result

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make HTTP request to CRW API.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: API path (e.g. /v1/scrape).
            json: Request body as dict.

        Returns:
            Parsed JSON response.

        Raises:
            requests.HTTPError: On 4xx/5xx responses.
        """
        url = f"{self.api_url}{path}"
        response = self._session.request(method, url, json=json)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_document(page: dict[str, Any]) -> Document:
        """Convert a CRW page response to a LangChain Document.

        Args:
            page: A single page dict from the CRW API response.

        Returns:
            A LangChain Document with content and metadata.
        """
        # Prefer markdown, fall back to other formats
        content = (
            page.get("markdown")
            or page.get("html")
            or page.get("rawHtml")
            or page.get("plainText")
            or ""
        )
        metadata = page.get("metadata", {})
        return Document(page_content=content, metadata=metadata)
