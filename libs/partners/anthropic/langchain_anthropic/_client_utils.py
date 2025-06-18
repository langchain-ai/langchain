from functools import lru_cache
from typing import Optional

import anthropic


@lru_cache
def _get_cached_client(
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    max_retries: int,
    timeout: Optional[float] = None,
) -> anthropic.Client:
    return anthropic.Client(
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )


@lru_cache
def _get_cached_async_client(
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    max_retries: int,
    timeout: Optional[float] = None,
) -> anthropic.AsyncClient:
    return anthropic.AsyncClient(
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )
