"""Helpers for creating OpenAI API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatOpenAI.

Logic is largely replicated from openai._base_client.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
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


# Thread-local storage for async httpx clients.
#
# Using @lru_cache (process-global) for the async client is unsafe when
# `ainvoke()` is called from multiple threads that each run their own event
# loop via `asyncio.run()`.  Each `asyncio.run()` call creates a *new* loop,
# then **closes** it on exit.  The cached client holds open connections bound
# to the first loop; when a later thread reuses the same cached client those
# connections try to communicate with a closed loop, raising:
#
#   RuntimeError: Event loop is closed
#   openai.APIConnectionError: Connection error.
#
# Storing one client per thread guarantees that the client is always used
# within the event loop that created it.
_thread_local_async_clients = threading.local()


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
    """Get default async httpx client.

    Uses a thread-local cache so that each thread (and therefore each asyncio
    event loop created by ``asyncio.run()``) gets its own client.  This avoids
    the ``RuntimeError: Event loop is closed`` / ``APIConnectionError`` that
    occurs when a process-global cached client is reused across event loops.

    If ``timeout`` is not hashable the client is built fresh (no caching).
    """
    try:
        cache_key = (base_url, timeout)
        hash(cache_key)
    except TypeError:
        return _build_async_httpx_client(base_url, timeout)

    if not hasattr(_thread_local_async_clients, "cache"):
        _thread_local_async_clients.cache = {}

    if cache_key not in _thread_local_async_clients.cache:
        _thread_local_async_clients.cache[cache_key] = _build_async_httpx_client(
            base_url, timeout
        )

    return _thread_local_async_clients.cache[cache_key]


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
