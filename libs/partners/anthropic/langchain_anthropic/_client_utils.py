"""Helpers for creating Anthropic API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatAnthropic.

Logic is largely replicated from anthropic._base_client.
"""

from __future__ import annotations

import asyncio
import os
import threading
from functools import lru_cache
from typing import Any

import anthropic

_NOT_GIVEN: Any = object()


class _SyncHttpxClientWrapper(anthropic.DefaultHttpxClient):
    """Borrowed from anthropic._base_client."""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncHttpxClientWrapper(anthropic.DefaultAsyncHttpxClient):
    """Borrowed from anthropic._base_client."""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass


@lru_cache
def _get_default_httpx_client(
    *,
    base_url: str | None,
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: str | None = None,
) -> _SyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com",
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    if anthropic_proxy is not None:
        kwargs["proxy"] = anthropic_proxy
    return _SyncHttpxClientWrapper(**kwargs)


def _build_async_httpx_client(
    *,
    base_url: str | None,
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: str | None = None,
) -> _AsyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com",
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    if anthropic_proxy is not None:
        kwargs["proxy"] = anthropic_proxy
    return _AsyncHttpxClientWrapper(**kwargs)


# Event-loop-aware cache for async httpx clients.
# An AsyncClient's underlying connections are bound to the event loop where
# they were first used.  If the cached client is reused from a *different*
# loop (e.g. after ``asyncio.run()`` creates a new loop), the stale
# connections raise ``RuntimeError: Event loop is closed``.
# We therefore key the cache on ``(base_url, timeout, proxy, loop_id)`` and
# discard entries whose loop has been closed.
_async_client_cache: dict[tuple, _AsyncHttpxClientWrapper] = {}
_async_client_cache_lock = threading.Lock()


def _get_default_async_httpx_client(
    *,
    base_url: str | None,
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: str | None = None,
) -> _AsyncHttpxClientWrapper:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    key = (base_url, timeout, anthropic_proxy, loop_id)

    with _async_client_cache_lock:
        client = _async_client_cache.get(key)
        if client is not None and not client.is_closed:
            return client

        # Evict entries whose event loop has closed.
        stale_keys = [
            k for k, v in _async_client_cache.items() if v.is_closed
        ]
        for k in stale_keys:
            del _async_client_cache[k]

        client = _build_async_httpx_client(
            base_url=base_url,
            timeout=timeout,
            anthropic_proxy=anthropic_proxy,
        )
        _async_client_cache[key] = client
        return client
