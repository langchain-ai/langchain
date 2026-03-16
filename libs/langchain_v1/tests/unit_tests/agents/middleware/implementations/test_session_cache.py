"""Unit tests for MiddlewareSessionCache."""

from __future__ import annotations

import threading
import time

import pytest

from langchain.agents.middleware.session_cache import MiddlewareSessionCache


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_put_and_get_returns_value() -> None:
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache()
    cache.put("s1", "hello")
    assert cache.get("s1") == "hello"


def test_get_missing_returns_none() -> None:
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache()
    assert cache.get("nonexistent") is None


def test_pop_returns_and_removes_value() -> None:
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache()
    cache.put("s1", "hello")
    result = cache.pop("s1")
    assert result == "hello"
    assert cache.get("s1") is None


def test_pop_missing_returns_none() -> None:
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache()
    assert cache.pop("nonexistent") is None


def test_len() -> None:
    cache: MiddlewareSessionCache[int] = MiddlewareSessionCache()
    assert len(cache) == 0
    cache.put("s1", 1)
    cache.put("s2", 2)
    assert len(cache) == 2
    cache.pop("s1")
    assert len(cache) == 1


def test_contains() -> None:
    cache: MiddlewareSessionCache[int] = MiddlewareSessionCache()
    cache.put("s1", 42)
    assert "s1" in cache
    assert "s2" not in cache


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


def test_sweep_expired_removes_stale_entries() -> None:
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache(idle_ttl=0.01)
    cache.put("s1", "value")
    time.sleep(0.05)
    evicted = cache.sweep_expired()
    assert len(evicted) == 1
    assert evicted[0] == "value"
    assert "s1" not in cache


def test_sweep_expired_keeps_fresh_entries() -> None:
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache(idle_ttl=60.0)
    cache.put("s1", "fresh")
    evicted = cache.sweep_expired()
    assert evicted == []
    assert "s1" in cache


def test_get_refreshes_ttl() -> None:
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache(idle_ttl=0.05)
    cache.put("s1", "value")
    time.sleep(0.03)
    cache.get("s1")  # refresh
    time.sleep(0.03)
    # total ~0.06 s since put, but only ~0.03 s since last get — should NOT expire
    evicted = cache.sweep_expired()
    assert evicted == []
    assert "s1" in cache


def test_sweep_expired_returns_empty_when_nothing_expired() -> None:
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache(idle_ttl=3600.0)
    cache.put("s1", "value")
    assert cache.sweep_expired() == []


# ---------------------------------------------------------------------------
# Eviction callback
# ---------------------------------------------------------------------------


def test_on_evict_called_on_sweep() -> None:
    evicted: list[str] = []
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache(
        idle_ttl=0.01, on_evict=evicted.append
    )
    cache.put("s1", "value")
    time.sleep(0.05)
    cache.sweep_expired()
    assert evicted == ["value"]


def test_on_evict_called_on_pop() -> None:
    evicted: list[str] = []
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache(
        on_evict=evicted.append
    )
    cache.put("s1", "value")
    cache.pop("s1")
    assert evicted == ["value"]


def test_on_evict_not_called_on_pop_missing() -> None:
    evicted: list[str] = []
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache(
        on_evict=evicted.append
    )
    cache.pop("nonexistent")
    assert evicted == []


def test_no_on_evict_sweep_does_not_raise() -> None:
    cache: MiddlewareSessionCache[str] = MiddlewareSessionCache(idle_ttl=0.01)
    cache.put("s1", "value")
    time.sleep(0.05)
    evicted = cache.sweep_expired()
    assert evicted == ["value"]


# ---------------------------------------------------------------------------
# Multiple sessions
# ---------------------------------------------------------------------------


def test_sessions_are_isolated() -> None:
    cache: MiddlewareSessionCache[int] = MiddlewareSessionCache()
    cache.put("s1", 1)
    cache.put("s2", 2)
    assert cache.get("s1") == 1
    assert cache.get("s2") == 2


def test_put_overwrites_existing_value() -> None:
    cache: MiddlewareSessionCache[int] = MiddlewareSessionCache()
    cache.put("s1", 1)
    cache.put("s1", 99)
    assert cache.get("s1") == 99


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_puts_are_safe() -> None:
    cache: MiddlewareSessionCache[int] = MiddlewareSessionCache()
    errors: list[Exception] = []

    def worker(session_id: str, value: int) -> None:
        try:
            for _ in range(100):
                cache.put(session_id, value)
                cache.get(session_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"s{i}", i)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_concurrent_sweep_and_put_are_safe() -> None:
    cache: MiddlewareSessionCache[int] = MiddlewareSessionCache(idle_ttl=0.001)
    errors: list[Exception] = []

    def putter() -> None:
        try:
            for i in range(200):
                cache.put(f"s{i}", i)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def sweeper() -> None:
        try:
            for _ in range(50):
                cache.sweep_expired()
                time.sleep(0.001)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=putter)
    t2 = threading.Thread(target=sweeper)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []
