"""Helpers for creating OpenAI API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatOpenAI.

Logic is largely replicated from openai._base_client.

Async client caching strategy
------------------------------
``httpx.AsyncClient`` instances are bound to the event loop that first issues a
request through them.  A process-global ``@lru_cache`` (the previous approach) shares
one client across all event loops, which causes ``httpcore.ConnectError`` /
``RuntimeError: Event loop is closed`` when a second call is made from a *different*
loop (e.g. two successive ``asyncio.run()`` calls in separate threads).

To fix this, async clients are cached in a ``weakref.WeakKeyDictionary`` keyed by the
*current* event loop.  The ``WeakKeyDictionary`` automatically removes the entry when
the loop is garbage-collected, so there is no unbounded memory growth.  Sync clients
are unaffected and continue to use a process-global ``@lru_cache``.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import weakref
from collections.abc import Awaitable, Callable
from functools import lru_cache
from threading import Lock
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


# Per-loop async client cache: {loop -> {(base_url, timeout) -> client}}.
# WeakKeyDictionary ensures entries are removed when the loop is garbage-collected.
_async_client_cache: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[tuple[str | None, Any], _AsyncHttpxClientWrapper],
] = weakref.WeakKeyDictionary()
_async_client_cache_lock = Lock()


def _cached_async_httpx_client(
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    """Return a cached async httpx client scoped to the current event loop.

    Using a process-global ``@lru_cache`` (the previous implementation) shares a
    single ``httpx.AsyncClient`` across all event loops.  When a second event loop
    runs after the first has been closed (e.g. two ``asyncio.run()`` calls in
    different threads), the cached client is bound to the dead loop and raises
    ``APIConnectionError``.

    This implementation caches one client *per event loop* so each loop always
    gets a fresh, compatible client.  The ``WeakKeyDictionary`` guarantees that
    cache entries are cleaned up automatically when a loop is garbage-collected.

    Args:
        base_url: Optional base URL override for the OpenAI API.
        timeout: Timeout configuration forwarded to ``httpx.AsyncClient``.

    Returns:
        An ``_AsyncHttpxClientWrapper`` bound to the current event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop — called during synchronous initialization.
        # Return a fresh (uncached) client; it will bind to whichever loop
        # first issues a request through it.
        return _build_async_httpx_client(base_url, timeout)

    cache_key = (base_url, timeout)
    with _async_client_cache_lock:
        if loop not in _async_client_cache:
            _async_client_cache[loop] = {}
        loop_clients = _async_client_cache[loop]
        if cache_key not in loop_clients:
            loop_clients[cache_key] = _build_async_httpx_client(base_url, timeout)
        return loop_clients[cache_key]


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
    """Get default httpx client.

    Uses cached client unless timeout is `httpx.Timeout`, which is not hashable.
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

    Because OpenAI and AsyncOpenAI clients support either sync or async callables for
    the API key, we need to resolve separate values here.
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
