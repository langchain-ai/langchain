"""Helpers for creating OpenAI API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatOpenAI.

Logic is largely replicated from openai._base_client.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import weakref
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any, cast

import openai
from pydantic import SecretStr


class _SyncHttpxClientWrapper(openai.DefaultHttpxClient):
    """Borrowed from openai._base_client."""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncHttpxClientWrapper(openai.DefaultAsyncHttpxClient):
    """Borrowed from openai._base_client."""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass


def _build_sync_httpx_client(
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    return _SyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )


def _build_async_httpx_client(
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    return _AsyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )


@lru_cache
def _cached_sync_httpx_client(
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    return _build_sync_httpx_client(base_url, timeout)


# Cache async httpx clients per event loop to avoid cross-loop connection reuse.
# httpx.AsyncClient connections are bound to the event loop they were created on.
# Using a process-global @lru_cache would cause 'Event loop is closed' errors when
# asyncio.run() is called multiple times (e.g. in multi-threaded environments, Celery
# workers, or sequential asyncio.run() calls), because each call creates and later
# closes a new event loop while the cached client still holds connections to the
# previous (now-closed) loop.
#
# Using WeakValueDictionary ensures that clients are automatically cleaned up when
# their associated event loop is garbage collected, preventing memory leaks.
_async_httpx_client_cache: weakref.WeakValueDictionary = (
    weakref.WeakValueDictionary()
)


def _cached_async_httpx_client(
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    """Get a cached async httpx client scoped to the current event loop.

    Unlike sync clients, async httpx clients cannot be safely shared across
    different event loops. This function uses the current loop's identity as
    part of the cache key, so each event loop gets its own client instance.
    """
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        # No running event loop; fall back to creating a fresh client.
        return _build_async_httpx_client(base_url, timeout)

    cache_key = (loop_id, base_url, timeout)
    client = _async_httpx_client_cache.get(cache_key)
    if client is None:
        client = _build_async_httpx_client(base_url, timeout)
        _async_httpx_client_cache[cache_key] = client
    return client


def _get_default_httpx_client(
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    """Get default httpx client.

    Uses cached client unless timeout is `httpx.Timeout`, which is not hashable.
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_sync_httpx_client(base_url, timeout)
    else:
        return _cached_sync_httpx_client(base_url, timeout)


def _get_default_async_httpx_client(
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    """Get default async httpx client, scoped to the current event loop.

    Async httpx clients are bound to the event loop they were created on, so
    they cannot be safely shared across different event loops. This function
    returns a client that is cached per-loop to avoid 'Event loop is closed'
    errors in multi-threaded or multi-loop environments.

    Uses a fresh (uncached) client when timeout is not hashable.
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_async_httpx_client(base_url, timeout)
    else:
        return _cached_async_httpx_client(base_url, timeout)


def _resolve_sync_and_async_api_keys(
    api_key: SecretStr | Callable[[], str] | Callable[[], Awaitable[str]],
) -> tuple[str | None | Callable[[], str], str | Callable[[], Awaitable[str]]]:
    """Resolve sync and async API key values.

    Because OpenAI and AsyncOpenAI clients support either sync or async callables
    for the API key, we need to resolve separate values here.
    """
    if isinstance(api_key, SecretStr):
        sync_api_key_value: str | None | Callable[[], str] = api_key.get_secret_value()
        async_api_key_value: str | Callable[[], Awaitable[str]] = (
            api_key.get_secret_value()
        )
    elif callable(api_key):
        if inspect.iscoroutinefunction(api_key):
            async_api_key_value = api_key
            sync_api_key_value = None
        else:
            sync_api_key_value = cast(Callable, api_key)

            async def async_api_key_wrapper() -> str:
                return await asyncio.get_event_loop().run_in_executor(
                    None, cast(Callable, api_key)
                )

            async_api_key_value = async_api_key_wrapper
    return sync_api_key_value, async_api_key_value
