"""Bocha HTTP client for API communication."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import aiohttp
import requests

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.bocha.cn/v1"
DEFAULT_TIMEOUT = 60


class BochaClient:
    """Low-level HTTP client for Bocha API endpoints.

    Handles authentication, request execution, and SSE stream parsing.
    All upper-layer classes (`ChatBocha`, `BochaSearchRun`, `BochaSearchResults`)
    share an instance of this client initialized via `initialize_client()`.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_API_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    # -- helpers ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Build standard authorization headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, endpoint: str) -> str:
        """Build the full URL for an endpoint."""
        if endpoint.startswith("http"):
            return endpoint
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    # -- sync methods -----------------------------------------------------

    def post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a synchronous POST request and return parsed JSON.

        Args:
            endpoint: The API endpoint path (e.g. ``/chat/completions``).
            payload: The JSON body to send.

        Returns:
            The parsed JSON response as a dictionary.

        Raises:
            ValueError: If the request fails after retries.
        """
        url = self._url(endpoint)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "Bocha API request failed (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                    )
        msg = (
            f"Error calling Bocha API after "
            f"{self.max_retries + 1} attempts: {last_error}"
        )
        raise ValueError(msg)

    def post_stream(
        self, endpoint: str, payload: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """Send a synchronous POST request and yield SSE events.

        Args:
            endpoint: The API endpoint path.
            payload: The JSON body to send.

        Yields:
            Parsed JSON dictionaries from each SSE ``data:`` line.
        """
        url = self._url(endpoint)
        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line:
                parsed = _parse_sse_line(line)
                if parsed is not None:
                    yield parsed

    # -- async methods ----------------------------------------------------

    async def apost(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send an asynchronous POST request and return parsed JSON.

        Args:
            endpoint: The API endpoint path.
            payload: The JSON body to send.

        Returns:
            The parsed JSON response as a dictionary.

        Raises:
            ValueError: If the request fails after retries.
        """
        url = self._url(endpoint)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        url,
                        headers=self._headers(),
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response,
                ):
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "Bocha API async request failed (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                    )
        msg = (
            f"Error calling Bocha API asynchronously "
            f"after {self.max_retries + 1} attempts: {last_error}"
        )
        raise ValueError(msg)

    async def apost_stream(
        self, endpoint: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Send an asynchronous POST request and yield SSE events.

        Args:
            endpoint: The API endpoint path.
            payload: The JSON body to send.

        Yields:
            Parsed JSON dictionaries from each SSE ``data:`` line.
        """
        url = self._url(endpoint)
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response,
        ):
            response.raise_for_status()
            async for line_bytes in response.content:
                line = line_bytes.decode("utf-8")
                parsed = _parse_sse_line(line)
                if parsed is not None:
                    yield parsed


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse a single SSE line into a dictionary.

    Args:
        line: The line to parse.

    Returns:
        A dictionary representation of the JSON data, or ``None``.
    """
    line = line.strip()
    if not line or not line.startswith("data: "):
        return None
    data = line[6:]
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return None
