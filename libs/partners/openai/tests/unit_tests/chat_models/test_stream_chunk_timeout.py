"""Unit tests for `_astream_with_chunk_timeout` and `StreamChunkTimeoutError`.

- Pass-through when items arrive in time.
- Timeout fires with a self-describing message + subclasses TimeoutError.
- Structured WARNING log carries `source=stream_chunk_timeout` +
    `timeout_s` so aggregate logging can distinguish app-layer from
    transport-layer timeouts.
- Source iterator's `aclose()` is called on early exit to release the
    underlying httpx connection promptly.
- Garbage in `LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` degrades safely.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typing_extensions import Self

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models._client_utils import (
    StreamChunkTimeoutError,
    _astream_with_chunk_timeout,
)

MODEL = "gpt-5.4"


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
    model = ChatOpenAI(model=MODEL)
    assert model.stream_chunk_timeout == 120.0


def test_stream_chunk_timeout_env_kill_switch_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env-var kill-switch: `_S=0` should disable the wrapper on the model."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "0")
    model = ChatOpenAI(model=MODEL)
    assert model.stream_chunk_timeout == 0.0


def test_stream_chunk_timeout_kwarg_none_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructor kwarg opt-out: `stream_chunk_timeout=None` persists."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = ChatOpenAI(model=MODEL, stream_chunk_timeout=None)
    assert model.stream_chunk_timeout is None


def test_stream_chunk_timeout_error_has_structured_attrs() -> None:
    """Structured payload mirrors the log `extra=`; no message-regex needed."""
    err = StreamChunkTimeoutError(0.5, model_name=MODEL, chunks_received=3)
    assert err.timeout_s == 0.5
    assert err.model_name == "gpt-5.4"
    assert err.chunks_received == 3
    text = str(err)
    assert "gpt-5.4" in text
    assert "chunks_received=3" in text


@pytest.mark.asyncio
async def test_astream_with_chunk_timeout_threads_model_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`model_name` flows into both the raised error and the structured log."""
    caplog.set_level(
        logging.WARNING, logger="langchain_openai.chat_models._client_utils"
    )
    source = _FakeSource(["a", "b"], per_item_sleep=0.2)
    with pytest.raises(StreamChunkTimeoutError) as exc_info:
        async for _ in _astream_with_chunk_timeout(
            source, 0.05, model_name="gpt-4o-mini"
        ):
            pass
    assert exc_info.value.model_name == "gpt-4o-mini"
    records = [
        r
        for r in caplog.records
        if getattr(r, "source", None) == "stream_chunk_timeout"
    ]
    assert records
    assert records[0].__dict__["model_name"] == "gpt-4o-mini"


def test_invalid_stream_chunk_timeout_env_emits_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fallback is logged at WARNING so the typo is discoverable."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "nonsense")
    caplog.set_level(
        logging.WARNING, logger="langchain_openai.chat_models._client_utils"
    )
    ChatOpenAI(model=MODEL)
    assert any(
        "LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_negative_stream_chunk_timeout_env_rejected(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Negative timeout typo must not silently disable the wrapper."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "-10")
    caplog.set_level(
        logging.WARNING, logger="langchain_openai.chat_models._client_utils"
    )
    model = ChatOpenAI(model=MODEL)
    assert model.stream_chunk_timeout == 120.0
    assert any(
        "negative" in r.getMessage().lower()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


class _SlowAsyncContextManager:
    """Async context manager that sleeps between streamed items."""

    def __init__(self, chunks: list[Any], per_item_sleep: float) -> None:
        self._chunks = list(chunks)
        self._sleep = per_item_sleep
        self._iter = iter(chunks)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Any:
        await asyncio.sleep(self._sleep)
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _SlowSyncContextManager:
    """Sync context manager mirror of `_SlowAsyncContextManager`.

    Sleeps between items in wall-clock time. The sync path never uses
    `asyncio.wait_for`, so a tight `stream_chunk_timeout` should have no
    effect here — that is the invariant we want to lock.
    """

    def __init__(self, chunks: list[Any], per_item_sleep: float) -> None:
        self._chunks = list(chunks)
        self._sleep = per_item_sleep
        self._iter = iter(chunks)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Any:
        import time as _time

        _time.sleep(self._sleep)
        try:
            return next(self._iter)
        except StopIteration:
            raise


@pytest.mark.asyncio
async def test_astream_integration_raises_stream_chunk_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: slow async stream + tight timeout must raise.

    Guards against a refactor that drops the `_astream_with_chunk_timeout`
    wrapper from the `_astream` path — unit tests on the helper alone
    wouldn't catch that regression.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = ChatOpenAI(model=MODEL, stream_chunk_timeout=0.05)
    fake_chunks = [
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hi"},
                    "finish_reason": None,
                }
            ],
        },
    ]
    mock_client = AsyncMock()

    async def mock_create(*args: Any, **kwargs: Any) -> _SlowAsyncContextManager:
        return _SlowAsyncContextManager(fake_chunks, per_item_sleep=0.3)

    mock_client.create = mock_create
    with (
        patch.object(llm, "async_client", mock_client),
        pytest.raises(StreamChunkTimeoutError) as exc_info,
    ):
        async for _ in llm.astream("hello"):
            pass
    assert exc_info.value.model_name == MODEL


def test_stream_sync_not_wrapped_by_chunk_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sync `llm.stream()` must not be subject to `stream_chunk_timeout`.

    Setting `stream_chunk_timeout=0.01` with a 100ms-per-chunk sync source
    would raise if the wrapper were (incorrectly) applied to the sync path.
    Completion without error proves the contract.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = ChatOpenAI(model=MODEL, stream_chunk_timeout=0.01)
    fake_chunks = [
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hi"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "c2",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"},
            ],
        },
    ]
    mock_client = MagicMock()

    def _create(*_args: Any, **_kwargs: Any) -> _SlowSyncContextManager:
        return _SlowSyncContextManager(fake_chunks, per_item_sleep=0.1)

    mock_client.create = _create
    with patch.object(llm, "client", mock_client):
        chunks = list(llm.stream("hello"))
    assert chunks, "sync stream should have delivered chunks"
