"""Unit tests for ``_astream_with_chunk_timeout`` and ``StreamChunkTimeoutError``.

Covers the P1 wall-clock per-chunk timeout wrapper:

- Pass-through when items arrive in time.
- Timeout fires with a self-describing message + subclasses TimeoutError.
- Structured WARNING log carries ``source=stream_chunk_timeout`` +
  ``timeout_s`` so aggregate logging can split P1 from P2 failures.
- Source iterator's ``aclose()`` is called on early exit (regression test
  for the connection-leak bug).
- Garbage in ``LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S`` degrades safely.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models._client_utils import (
    StreamChunkTimeoutError,
    _astream_with_chunk_timeout,
)


class _FakeSource:
    """AsyncIterator with an observable aclose() for leak-testing."""

    def __init__(self, items: list[Any], per_item_sleep: float = 0.0) -> None:
        self._items = list(items)
        self._sleep = per_item_sleep
        self.aclose_count = 0

    def __aiter__(self) -> _FakeSource:
        return self

    async def __anext__(self) -> Any:
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)

    async def aclose(self) -> None:
        self.aclose_count += 1


@pytest.mark.asyncio
async def test_astream_with_chunk_timeout_passes_through() -> None:
    """Fast source + generous timeout: every item should be delivered."""
    source = _FakeSource(["a", "b", "c"], per_item_sleep=0.0)
    collected = [item async for item in _astream_with_chunk_timeout(source, 5.0)]
    assert collected == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_astream_with_chunk_timeout_disabled_passes_through() -> None:
    """timeout=None / timeout=0 disables the bound; still iterates normally."""
    source_none = _FakeSource(["a", "b"])
    collected_none = [
        item async for item in _astream_with_chunk_timeout(source_none, None)
    ]
    assert collected_none == ["a", "b"]

    source_zero = _FakeSource(["x", "y"])
    collected_zero = [
        item async for item in _astream_with_chunk_timeout(source_zero, 0.0)
    ]
    assert collected_zero == ["x", "y"]


@pytest.mark.asyncio
async def test_astream_with_chunk_timeout_fires() -> None:
    """Slow source + tight timeout: ``StreamChunkTimeoutError`` fires."""
    source = _FakeSource(["a", "b"], per_item_sleep=0.2)
    with pytest.raises(StreamChunkTimeoutError) as exc_info:
        async for _ in _astream_with_chunk_timeout(source, 0.05):
            pass

    # Backward-compat: existing `except TimeoutError:` handlers must still catch.
    assert isinstance(exc_info.value, asyncio.TimeoutError)
    assert isinstance(exc_info.value, TimeoutError)

    # Self-describing message names the knob and env var so operators can act.
    msg = str(exc_info.value)
    assert "stream_chunk_timeout" in msg
    assert "LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S" in msg


@pytest.mark.asyncio
async def test_astream_with_chunk_timeout_logs_on_fire(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Structured log carries source + timeout_s for aggregate-log filtering."""
    # Pin the logger + level; don't rely on caplog's default or module
    # inheritance so the test can't silently no-op.
    caplog.set_level(
        logging.WARNING, logger="langchain_openai.chat_models._client_utils"
    )

    source = _FakeSource(["a"], per_item_sleep=0.2)
    with pytest.raises(StreamChunkTimeoutError):
        async for _ in _astream_with_chunk_timeout(source, 0.05):
            pass

    records = [
        r
        for r in caplog.records
        if r.name == "langchain_openai.chat_models._client_utils"
        and getattr(r, "source", None) == "stream_chunk_timeout"
    ]
    assert len(records) == 1, f"expected one structured record, got {len(records)}"
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.__dict__["timeout_s"] == 0.05


@pytest.mark.asyncio
async def test_astream_with_chunk_timeout_closes_source_on_early_exit() -> None:
    """Regression test for the connection-leak bug.

    If the timeout fires, we must ``aclose()`` the underlying iterator so the
    httpx streaming connection is released promptly; ditto if a consumer
    explicitly closes our wrapper.
    """
    # Case 1: timeout fires -> aclose() propagates.
    timed_out_source = _FakeSource(["a"], per_item_sleep=0.2)
    with pytest.raises(StreamChunkTimeoutError):
        async for _ in _astream_with_chunk_timeout(timed_out_source, 0.05):
            pass
    assert timed_out_source.aclose_count == 1

    # Case 2: consumer explicitly closes the wrapper after one yield.
    closer_source = _FakeSource(["a", "b", "c"], per_item_sleep=0.0)
    # Cast to AsyncGenerator so mypy sees the aclose() method; the helper
    # is always implemented as an async generator at runtime.
    wrapper = cast(
        "AsyncGenerator[Any, None]",
        _astream_with_chunk_timeout(closer_source, 5.0),
    )
    got = await wrapper.__anext__()
    assert got == "a"
    await wrapper.aclose()
    assert closer_source.aclose_count == 1


def test_invalid_stream_chunk_timeout_env_degrades_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garbage env var -> model init succeeds with the 120s default."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "not-a-float")
    model = ChatOpenAI(model="gpt-4o")
    assert model.stream_chunk_timeout == 120.0
