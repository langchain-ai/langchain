"""Helpers for creating OpenAI API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatOpenAI.

Logic is largely replicated from openai._base_client.
"""

import asyncio
import os
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
        except Exception:
            pass


class _AsyncHttpxClientWrapper(openai.DefaultAsyncHttpxClient):
    """Borrowed from openai._base_client"""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:
            pass


@lru_cache
def _get_default_httpx_client(
    base_url: Optional[str], timeout: Any
) -> _SyncHttpxClientWrapper:
    return _SyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )


@lru_cache
def _get_default_async_httpx_client(
    base_url: Optional[str], timeout: Any
) -> _AsyncHttpxClientWrapper:
    return _AsyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )
