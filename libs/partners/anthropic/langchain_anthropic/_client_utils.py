"""Helpers for creating Anthropic API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatAnthropic.

Logic is largely replicated from anthropic._base_client.
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Any, Optional

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
    base_url: Optional[str],
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: Optional[str] = None,
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


@lru_cache
def _get_default_async_httpx_client(
    *,
    base_url: Optional[str],
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: Optional[str] = None,
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
