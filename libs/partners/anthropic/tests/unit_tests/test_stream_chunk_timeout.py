"""Unit tests for `_astream_with_chunk_timeout` and `StreamChunkTimeoutError`.

- Pass-through when items arrive in time.
- Timeout fires with a self-describing message + subclasses TimeoutError.
- Structured WARNING log carries `source=stream_chunk_timeout` +
    `timeout_s` so aggregate logging can distinguish app-layer from
    transport-layer timeouts.
- Source iterator's `aclose()` is called on early exit to release the
    underlying httpx connection promptly.
- Garbage in `LANGCHAIN_ANTHROPIC_STREAM_CHUNK_TIMEOUT_S` degrades safely.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import patch

import pytest

from langchain_anthropic import ChatAnthropic
from langchain_anthropic._client_utils import (
    StreamChunkTimeoutError,
    _astream_with_chunk_timeout,
)

MODEL = "claude-opus-4-7"


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


async def test_astream_with_chunk_timeout_passes_through() -> None:
    """Fast source + generous timeout: every item should be delivered."""
    source = _FakeSource(["a", "b", "c"], per_item_sleep=0.0)
    collected = [item async for item in _astream_with_chunk_timeout(source, 5.0)]
    assert collected == ["a", "b", "c"]


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


async def test_astream_with_chunk_timeout_fires() -> None:
    """Slow source + tight timeout: `StreamChunkTimeoutError` fires."""
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
    assert "LANGCHAIN_ANTHROPIC_STREAM_CHUNK_TIMEOUT_S" in msg


async def test_astream_with_chunk_timeout_logs_on_fire(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Structured log carries source + timeout_s for aggregate-log filtering."""
    caplog.set_level(logging.WARNING, logger="langchain_anthropic._client_utils")

    source = _FakeSource(["a"], per_item_sleep=0.2)
    with pytest.raises(StreamChunkTimeoutError):
        async for _ in _astream_with_chunk_timeout(source, 0.05):
            pass

    records = [
        r
        for r in caplog.records
        if r.name == "langchain_anthropic._client_utils"
        and getattr(r, "source", None) == "stream_chunk_timeout"
    ]
    assert len(records) == 1, f"expected one structured record, got {len(records)}"
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.__dict__["timeout_s"] == 0.05


async def test_astream_with_chunk_timeout_closes_source_on_early_exit() -> None:
    """aclose() is called on early exit so the httpx connection is released promptly.

    Covers both the timeout-fires path and the consumer-closes-wrapper path.
    """
    # Case 1: timeout fires -> aclose() propagates.
    timed_out_source = _FakeSource(["a"], per_item_sleep=0.2)
    with pytest.raises(StreamChunkTimeoutError):
        async for _ in _astream_with_chunk_timeout(timed_out_source, 0.05):
            pass
    assert timed_out_source.aclose_count == 1

    # Case 2: consumer explicitly closes the wrapper after one yield.
    closer_source = _FakeSource(["a", "b", "c"], per_item_sleep=0.0)
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_STREAM_CHUNK_TIMEOUT_S", "not-a-float")
    model = ChatAnthropic(model=MODEL)
    assert model.stream_chunk_timeout == 120.0


def test_stream_chunk_timeout_env_kill_switch_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env-var kill-switch: `_S=0` should disable the wrapper on the model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_STREAM_CHUNK_TIMEOUT_S", "0")
    model = ChatAnthropic(model=MODEL)
    assert model.stream_chunk_timeout == 0.0


def test_stream_chunk_timeout_kwarg_none_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructor kwarg opt-out: `stream_chunk_timeout=None` persists."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    model = ChatAnthropic(model=MODEL, stream_chunk_timeout=None)
    assert model.stream_chunk_timeout is None


def test_stream_chunk_timeout_error_has_structured_attrs() -> None:
    """Structured payload mirrors the log `extra=`; no message-regex needed."""
    err = StreamChunkTimeoutError(0.5, model_name=MODEL, chunks_received=3)
    assert err.timeout_s == 0.5
    assert err.model_name == MODEL
    assert err.chunks_received == 3
    text = str(err)
    assert MODEL in text
    assert "chunks_received=3" in text


async def test_astream_with_chunk_timeout_threads_model_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`model_name` flows into both the raised error and the structured log."""
    caplog.set_level(logging.WARNING, logger="langchain_anthropic._client_utils")
    source = _FakeSource(["a", "b"], per_item_sleep=0.2)
    with pytest.raises(StreamChunkTimeoutError) as exc_info:
        async for _ in _astream_with_chunk_timeout(
            source, 0.05, model_name="claude-3-5-sonnet"
        ):
            pass
    assert exc_info.value.model_name == "claude-3-5-sonnet"
    records = [
        r
        for r in caplog.records
        if getattr(r, "source", None) == "stream_chunk_timeout"
    ]
    assert records
    assert records[0].__dict__["model_name"] == "claude-3-5-sonnet"


def test_invalid_stream_chunk_timeout_env_emits_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fallback is logged at WARNING so the typo is discoverable."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_STREAM_CHUNK_TIMEOUT_S", "nonsense")
    caplog.set_level(logging.WARNING, logger="langchain_anthropic._client_utils")
    ChatAnthropic(model=MODEL)
    assert any(
        "LANGCHAIN_ANTHROPIC_STREAM_CHUNK_TIMEOUT_S" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_negative_stream_chunk_timeout_env_rejected(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Negative timeout typo must not silently disable the wrapper."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_STREAM_CHUNK_TIMEOUT_S", "-10")
    caplog.set_level(logging.WARNING, logger="langchain_anthropic._client_utils")
    model = ChatAnthropic(model=MODEL)
    assert model.stream_chunk_timeout == 120.0
    assert any(
        "negative" in r.getMessage().lower()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_negative_stream_chunk_timeout_kwarg_rejected(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Negative kwarg (e.g., from YAML/JSON configs) must not disable the wrapper."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    caplog.set_level(logging.WARNING, logger="langchain_anthropic.chat_models")
    model = ChatAnthropic(model=MODEL, stream_chunk_timeout=-10)
    assert model.stream_chunk_timeout == 120.0
    assert any(
        "negative" in r.getMessage().lower()
        and "stream_chunk_timeout" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_zero_stream_chunk_timeout_kwarg_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`stream_chunk_timeout=0` is the documented opt-out and must persist."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    model = ChatAnthropic(model=MODEL, stream_chunk_timeout=0)
    assert model.stream_chunk_timeout == 0


class _SlowAsyncStream:
    """Async iterator that sleeps between streamed events."""

    def __init__(self, events: list[Any], per_item_sleep: float) -> None:
        self._events = iter(events)
        self._sleep = per_item_sleep

    def __aiter__(self) -> _SlowAsyncStream:
        return self

    async def __anext__(self) -> Any:
        await asyncio.sleep(self._sleep)
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


async def test_astream_integration_raises_stream_chunk_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: slow async stream + tight timeout must raise.

    Guards against a refactor that drops the `_astream_with_chunk_timeout`
    wrapper from the `_astream` path — unit tests on the helper alone
    wouldn't catch that regression.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    llm = ChatAnthropic(model=MODEL, stream_chunk_timeout=0.05)

    # A single dummy event that never arrives in time (sleep > timeout).
    fake_events: list[Any] = [
        {"type": "message_start", "message": {"model": MODEL, "usage": {}}},
    ]

    async def fake_acreate(payload: dict) -> _SlowAsyncStream:
        return _SlowAsyncStream(fake_events, per_item_sleep=0.3)

    with (
        patch.object(llm, "_acreate", side_effect=fake_acreate),
        pytest.raises(StreamChunkTimeoutError) as exc_info,
    ):
        async for _ in llm._astream([]):
            pass
    assert exc_info.value.model_name == MODEL


async def test_astream_integration_passes_through_when_timeout_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `stream_chunk_timeout=None` a slow stream must not raise."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    llm = ChatAnthropic(model=MODEL, stream_chunk_timeout=None)

    fake_events: list[Any] = []  # no events -> clean stop

    async def fake_acreate(payload: dict) -> _SlowAsyncStream:
        return _SlowAsyncStream(fake_events, per_item_sleep=0.0)

    with patch.object(llm, "_acreate", side_effect=fake_acreate):
        collected = [chunk async for chunk in llm._astream([])]
    assert collected == []
