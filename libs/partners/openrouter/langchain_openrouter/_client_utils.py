"""Cached httpx client factories for langchain-openrouter.

Mirrors the pattern from langchain-openai's ``_client_utils``
(originally PR #32531) to avoid per-instance httpx client creation.

Without this cache, each ``ChatOpenRouter(...)`` constructs a fresh
``httpx.Client`` / ``httpx.AsyncClient`` pair. When instances are created
per-request (LangGraph factory graphs, FastAPI DI, etc.), the discarded
clients hold onto TLS connections until interpreter exit, causing linear
socket growth and eventual resource exhaustion.  See
``langchain-openai``'s ``_client_utils`` for the detailed analogue.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx


class _SyncClientWrapper(httpx.Client):
    """Wraps a sync httpx client for safe __del__ cleanup.

    Borrowed from ``httpx``'s own ``DefaultHttpxClient`` pattern.
    """

    def __del__(self) -> None:
        if self.is_closed:
            return
        try:
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncClientWrapper(httpx.AsyncClient):
    """Wraps an async httpx client for safe __del__ cleanup."""

    def __del__(self) -> None:
        if self.is_closed:
            return
        try:
            import asyncio

            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass


def _build_sync_client(
    base_url: str | None,
    timeout: Any,
    headers: dict[str, str] | None = None,
) -> _SyncClientWrapper:
    """Build a sync httpx client with the given parameters."""
    kwargs: dict[str, Any] = {"timeout": timeout, "follow_redirects": True}
    if base_url:
        kwargs["base_url"] = base_url
    if headers:
        kwargs["headers"] = headers
    return _SyncClientWrapper(**kwargs)


def _build_async_client(
    base_url: str | None,
    timeout: Any,
    headers: dict[str, str] | None = None,
) -> _AsyncClientWrapper:
    """Build an async httpx client with the given parameters."""
    kwargs: dict[str, Any] = {"timeout": timeout, "follow_redirects": True}
    if base_url:
        kwargs["base_url"] = base_url
    if headers:
        kwargs["headers"] = headers
    return _AsyncClientWrapper(**kwargs)


def _cache_key(
    base_url: str | None,
    timeout: Any,
    headers_tuple: tuple[tuple[str, str], ...],
) -> tuple[str | None, Any, tuple[tuple[str, str], ...]]:
    """Build a stable hash cache key for client parameters."""
    return (base_url, timeout, headers_tuple)


@lru_cache
def _cached_sync_client(
    base_url: str | None,
    timeout: Any,
    headers_tuple: tuple[tuple[str, str], ...],
) -> _SyncClientWrapper:
    """Build a cached sync httpx client.

    ``headers_tuple`` is used instead of ``dict`` because dicts are not
    hashable and cannot be passed to ``@lru_cache``.
    """
    headers = dict(headers_tuple) if headers_tuple else None
    return _build_sync_client(base_url, timeout, headers)


@lru_cache
def _cached_async_client(
    base_url: str | None,
    timeout: Any,
    headers_tuple: tuple[tuple[str, str], ...],
) -> _AsyncClientWrapper:
    """Build a cached async httpx client."""
    headers = dict(headers_tuple) if headers_tuple else None
    return _build_async_client(base_url, timeout, headers)


def get_default_httpx_client(
    *,
    base_url: str | None = None,
    timeout: Any = httpx.USE_DEFAULT,
    headers: dict[str, str] | None = None,
) -> httpx.Client:
    """Get or create a default sync httpx client.

    Uses an LRU cache so that identically-configured callers share a single
    connection pool.  Pass ``headers`` to include custom per-request headers
    (e.g. OpenRouter attribution headers).

    Args:
        base_url: Optional base URL for the client.
        timeout: Timeout configuration (default: httpx.USE_DEFAULT).
        headers: Optional extra headers to include on every request.

    Returns:
        An httpx.Client instance (potentially shared).
    """
    headers_tuple = tuple(sorted(headers.items())) if headers else ()
    try:
        hash(timeout)
    except TypeError:
        # Unhashable timeout (e.g. httpx.Timeout with custom values) --
        # fall through to an uncached build.
        return _build_sync_client(base_url, timeout, headers)
    return _cached_sync_client(base_url, timeout, headers_tuple)


def get_default_async_httpx_client(
    *,
    base_url: str | None = None,
    timeout: Any = httpx.USE_DEFAULT,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """Get or create a default async httpx client.

    Same caching semantics as ``get_default_httpx_client``.

    Args:
        base_url: Optional base URL for the client.
        timeout: Timeout configuration.
        headers: Optional extra headers.

    Returns:
        An httpx.AsyncClient instance (potentially shared).
    """
    headers_tuple = tuple(sorted(headers.items())) if headers else ()
    try:
        hash(timeout)
    except TypeError:
        return _build_async_client(base_url, timeout, headers)
    return _cached_async_client(base_url, timeout, headers_tuple)
