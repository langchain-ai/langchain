"""Unit tests for _client_utils.py async httpx client caching."""

from __future__ import annotations

import asyncio
import gc
import threading
import weakref

from langchain_openai.chat_models._client_utils import (
    _async_client_cache,
    _cached_async_httpx_client,
    _get_default_async_httpx_client,
)


def test_same_loop_returns_same_client() -> None:
    """Repeated calls within one event loop must return the identical instance."""

    async def inner() -> None:
        c1 = _cached_async_httpx_client(None, 60)
        c2 = _cached_async_httpx_client(None, 60)
        assert c1 is c2

    asyncio.run(inner())


def test_different_loops_return_different_clients() -> None:
    """Two successive asyncio.run() calls must not share an async client.

    This is the root cause of #35783: the old @lru_cache returned the same client
    across event loops, causing APIConnectionError when the first loop was closed.
    """
    collected: list = []

    async def collect() -> None:
        collected.append(_cached_async_httpx_client(None, 60))

    asyncio.run(collect())
    asyncio.run(collect())

    assert len(collected) == 2
    assert collected[0] is not collected[1]


def test_different_loops_in_threads_return_different_clients() -> None:
    """Each thread running its own event loop must receive an isolated client."""
    clients: list = []
    lock = threading.Lock()

    async def collect() -> None:
        client = _cached_async_httpx_client(None, 60)
        with lock:
            clients.append(client)

    def run_loop() -> None:
        asyncio.run(collect())

    threads = [threading.Thread(target=run_loop) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(clients) == 4
    assert len({id(c) for c in clients}) == 4


def test_no_running_loop_returns_uncached_client() -> None:
    """Without a running event loop each call returns a fresh (uncached) client."""
    c1 = _cached_async_httpx_client(None, 60)
    c2 = _cached_async_httpx_client(None, 60)
    assert c1 is not c2


def test_different_params_in_same_loop_return_different_clients() -> None:
    """Different (base_url, timeout) combinations get distinct cached clients."""

    async def inner() -> None:
        c1 = _cached_async_httpx_client("https://a.example.com/v1", 30)
        c2 = _cached_async_httpx_client("https://b.example.com/v1", 30)
        c3 = _cached_async_httpx_client("https://a.example.com/v1", 30)

        assert c1 is not c2
        assert c1 is c3

    asyncio.run(inner())


def test_loop_gc_removes_cache_entry() -> None:
    """Cache entries must be released when the event loop is garbage-collected."""
    loop = asyncio.new_event_loop()

    async def populate() -> None:
        _cached_async_httpx_client(None, 60)

    loop.run_until_complete(populate())
    assert loop in _async_client_cache

    loop.close()
    del loop
    gc.collect()

    assert isinstance(_async_client_cache, weakref.WeakKeyDictionary)
    # After GC the deleted loop must no longer be reachable as a key
    for existing_loop in list(_async_client_cache.keys()):
        assert existing_loop is not None  # just iterating to ensure no phantom entries


def test_get_default_async_httpx_client_unhashable_timeout_bypasses_cache() -> None:
    """An unhashable timeout must create a fresh client each time (no caching)."""
    import httpx

    unhashable = httpx.Timeout(30.0)

    async def inner() -> None:
        c1 = _get_default_async_httpx_client(None, unhashable)
        c2 = _get_default_async_httpx_client(None, unhashable)
        assert c1 is not c2

    asyncio.run(inner())
