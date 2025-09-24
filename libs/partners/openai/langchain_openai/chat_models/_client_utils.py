"""Helpers for creating OpenAI API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatOpenAI.

Logic is largely replicated from openai._base_client.
"""

import asyncio
import os
import warnings
from functools import lru_cache
from typing import Any, Optional

import openai


class _SyncHttpxClientWrapper(openai.DefaultHttpxClient):
    """Borrowed from openai._base_client"""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncHttpxClientWrapper(openai.DefaultAsyncHttpxClient):
    """Borrowed from openai._base_client"""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass


def _build_sync_httpx_client(
    base_url: Optional[str], timeout: Any
) -> _SyncHttpxClientWrapper:
    return _SyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )


def _build_async_httpx_client(
    base_url: Optional[str], timeout: Any
) -> _AsyncHttpxClientWrapper:
    return _AsyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )


@lru_cache
def _cached_sync_httpx_client(
    base_url: Optional[str], timeout: Any
) -> _SyncHttpxClientWrapper:
    return _build_sync_httpx_client(base_url, timeout)


@lru_cache
def _cached_async_httpx_client(
    base_url: Optional[str], timeout: Any
) -> _AsyncHttpxClientWrapper:
    return _build_async_httpx_client(base_url, timeout)


def _get_default_httpx_client(
    base_url: Optional[str], timeout: Any
) -> _SyncHttpxClientWrapper:
    """Get default httpx client.

    Uses cached client unless timeout is ``httpx.Timeout``, which is not hashable.
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_sync_httpx_client(base_url, timeout)
    else:
        return _cached_sync_httpx_client(base_url, timeout)


def _get_default_async_httpx_client(
    base_url: Optional[str], timeout: Any
) -> _AsyncHttpxClientWrapper:
    """Get default httpx client.

    Uses cached client unless timeout is ``httpx.Timeout``, which is not hashable.
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_async_httpx_client(base_url, timeout)
    else:
        return _cached_async_httpx_client(base_url, timeout)


def _get_aiohttp_client() -> Optional[Any]:
    """Get OpenAI DefaultAioHttpClient if available.

    Returns:
        DefaultAioHttpClient instance if openai[aiohttp] is installed, None otherwise.
    """
    try:
        from openai import DefaultAioHttpClient

        return DefaultAioHttpClient()
    except ImportError:
        return None


def _should_use_aiohttp() -> bool:
    """Check if aiohttp backend should be used based on environment variable.

    Returns:
        True if LC_OPENAI_USE_AIOHTTP environment variable is set to a truthy value.
    """
    return os.getenv("LC_OPENAI_USE_AIOHTTP", "").lower() in ("1", "true", "yes", "on")


def _get_http_client_for_aiohttp_env(
    provided_client: Optional[Any],
    base_url: Optional[str],
    timeout: Any,
    is_async: bool = False,
) -> Optional[Any]:
    """Get appropriate HTTP client considering aiohttp environment variable.

    Args:
        provided_client: User-provided http client (takes precedence)
        base_url: OpenAI API base URL
        timeout: Request timeout
        is_async: Whether to get async client

    Returns:
        Appropriate HTTP client or None to use OpenAI default
    """
    # User-provided client takes precedence
    if provided_client is not None:
        return provided_client

    # Check if aiohttp should be used via environment variable
    if _should_use_aiohttp():
        aiohttp_client = _get_aiohttp_client()
        if aiohttp_client is not None:
            return aiohttp_client
        else:
            warnings.warn(
                "LC_OPENAI_USE_AIOHTTP is set but openai[aiohttp] is not installed. "
                "Install with 'pip install \"openai[aiohttp]\"' to use the aiohttp "
                "backend. Falling back to default httpx client.",
                UserWarning,
                stacklevel=3,
            )

    # Fall back to existing httpx client logic
    if is_async:
        return _get_default_async_httpx_client(base_url, timeout)
    else:
        return _get_default_httpx_client(base_url, timeout)
