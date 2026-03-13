"""Helpers for creating Anthropic API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatAnthropic.

Logic is largely replicated from anthropic._base_client.
"""

from __future__ import annotations

import asyncio
import os
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


# Async httpx clients are cached per event loop to avoid sharing connections
# across loops, which causes RuntimeError("Event loop is closed").
# See: https://github.com/langchain-ai/langchain/issues/35783
_async_httpx_client_cache: dict[
    tuple[str | None, Any, str | None, int], _AsyncHttpxClientWrapper
] = {}


def _get_default_async_httpx_client(
    *,
    base_url: str | None,
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: str | None = None,
) -> _AsyncHttpxClientWrapper:
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        # No running loop — return a fresh client (don't cache).
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

    key = (base_url, timeout, anthropic_proxy, loop_id)
    client = _async_httpx_client_cache.get(key)
    if client is not None and not client.is_closed:
        return client

    # Clean up stale entries for dead loops before adding a new one.
    stale_keys = [k for k, v in _async_httpx_client_cache.items() if v.is_closed]
    for k in stale_keys:
        del _async_httpx_client_cache[k]

    kwargs = {
        "base_url": base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com",
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    if anthropic_proxy is not None:
        kwargs["proxy"] = anthropic_proxy
    client = _AsyncHttpxClientWrapper(**kwargs)
    _async_httpx_client_cache[key] = client
    return client
