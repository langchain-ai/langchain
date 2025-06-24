"""Helpers for creating Anthropic API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatAnthropic.

Logic is largely replicated from anthropic._base_client.
"""

import asyncio
import os
from functools import lru_cache
from typing import Any, Optional
from weakref import WeakKeyDictionary

import anthropic

_NOT_GIVEN: Any = object()


class _SyncHttpxClientWrapper(anthropic.DefaultHttpxClient):
    """Borrowed from anthropic._base_client"""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            self.close()
        except Exception:
            pass


class _AsyncHttpxClientWrapper(anthropic.DefaultAsyncHttpxClient):
    """Borrowed from anthropic._base_client"""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:
            pass


# Cache sync client
@lru_cache
def _get_default_httpx_client(
    *,
    base_url: Optional[str],
    timeout: Any = _NOT_GIVEN,
) -> _SyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com",
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    return _SyncHttpxClientWrapper(**kwargs)


# Cache async client - must store caches per event loop
_client_caches: WeakKeyDictionary = WeakKeyDictionary()


def _create_async_httpx_client(
    *,
    base_url: Optional[str],
    timeout: Any = _NOT_GIVEN,
) -> _AsyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com",
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    return _AsyncHttpxClientWrapper(**kwargs)


def _get_default_async_httpx_client(
    *,
    base_url: Optional[str],
    timeout: Any = _NOT_GIVEN,
) -> _AsyncHttpxClientWrapper:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If no event loop is running, don't cache
        return _create_async_httpx_client(base_url=base_url, timeout=timeout)

    # Get or create cache for this event loop
    if loop not in _client_caches:
        _client_caches[loop] = {}
    cache_key = (base_url, timeout)
    if cache_key not in _client_caches[loop]:
        _client_caches[loop][cache_key] = _create_async_httpx_client(
            base_url=base_url, timeout=timeout
        )

    return _client_caches[loop][cache_key]
