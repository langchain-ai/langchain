from functools import lru_cache
from typing import Any, Optional

import anthropic

_NOT_GIVEN: Any = object()


@lru_cache
def _get_cached_client(
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    max_retries: int,
    timeout: Any = _NOT_GIVEN,
) -> anthropic.Client:
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "max_retries": max_retries,
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout

    return anthropic.Client(**kwargs)


@lru_cache
def _get_cached_async_client(
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    max_retries: int,
    timeout: Any = _NOT_GIVEN,
) -> anthropic.AsyncClient:
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "max_retries": max_retries,
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout

    return anthropic.AsyncClient(**kwargs)
