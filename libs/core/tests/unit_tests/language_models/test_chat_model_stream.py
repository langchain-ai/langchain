"""Tests for ChatModelStream, AsyncChatModelStream, and projections."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    AsyncProjection,
    ChatModelStream,
    SyncProjection,
    SyncTextProjection,
    dispatch_event,
)

if TYPE_CHECKING:
    from langchain_protocol.protocol import MessagesData

# ---------------------------------------------------------------------------
# Projection unit tests
# ---------------------------------------------------------------------------


class TestSyncProjection:
    """Test SyncProjection push/pull mechanics."""

    def test_push_and_iterate(self) -> None:
        proj = SyncProjection()
        proj._push("a")
        proj._push("b")
        proj._finish(["a", "b"])
        assert list(proj) == ["a", "b"]

    def test_get_returns_final_value(self) -> None:
        proj = SyncProjection()
        proj._push("x")
        proj._finish("final")
        assert proj.get() == "final"

    def test_request_more_pulls(self) -> None:
        proj = SyncProjection()
        calls = iter(["a", "b", None])

        def pump() -> bool:
            val = next(calls)
            if val is None:
                proj._finish("ab")
                return True
            proj._push(val)
            return True

        proj._request_more = pump
        assert list(proj) == ["a", "b"]
        assert proj.get() == "ab"

    def test_error_propagation(self) -> None:
        proj = SyncProjection()
        proj._push("partial")
        proj._fail(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            list(proj)

    def test_error_on_get(self) -> None:
        proj = SyncProjection()
        proj._fail(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            proj.get()

    def test_multi_cursor_replay(self) -> None:
        proj = SyncProjection()
        proj._push("a")
        proj._push("b")
        proj._finish(None)
        assert list(proj) == ["a", "b"]
        assert list(proj) == ["a", "b"]  # Second iteration replays

    def test_empty_projection(self) -> None:
        proj = SyncProjection()
        proj._finish([])
        assert list(proj) == []
        assert proj.get() == []


class TestSyncTextProjection:
    """Test SyncTextProjection string convenience methods."""

    def test_str_drains(self) -> None:
        proj = SyncTextProjection()
        proj._push("Hello")
        proj._push(" world")
        proj._finish("Hello world")
        assert str(proj) == "Hello world"

    def test_str_with_pump(self) -> None:
        proj = SyncTextProjection()
        done = False

        def pump() -> bool:
            nonlocal done
            if not done:
                proj._push("Hi")
                proj._finish("Hi")
                done = True
                return True
            return False

        proj._request_more = pump
        assert str(proj) == "Hi"

    def test_bool_nonempty(self) -> None:
        proj = SyncTextProjection()
        assert not proj
        proj._push("x")
        assert proj

    def test_repr(self) -> None:
        proj = SyncTextProjection()
        proj._push("hello")
        assert repr(proj) == "'hello'"
        proj._finish("hello")
        assert repr(proj) == "'hello'"


class TestAsyncProjection:
    """Test AsyncProjection async iteration and awaiting."""

    @pytest.mark.asyncio
    async def test_await_final_value(self) -> None:
        proj = AsyncProjection()
        proj._push("a")
        proj._finish("final")
        assert await proj == "final"

    @pytest.mark.asyncio
    async def test_async_iter(self) -> None:
        proj = AsyncProjection()

        async def produce() -> None:
            await asyncio.sleep(0)
            proj._push("x")
            await asyncio.sleep(0)
            proj._push("y")
            await asyncio.sleep(0)
            proj._finish("xy")

        asyncio.get_running_loop().create_task(produce())
        deltas = [d async for d in proj]
        assert deltas == ["x", "y"]

    @pytest.mark.asyncio
    async def test_error_on_await(self) -> None:
        proj = AsyncProjection()
        proj._fail(ValueError("async boom"))
        with pytest.raises(ValueError, match="async boom"):
            await proj

    @pytest.mark.asyncio
    async def test_error_on_iter(self) -> None:
        proj = AsyncProjection()
        proj._push("partial")
        proj._fail(ValueError("mid-stream"))
        with pytest.raises(ValueError, match="mid-stream"):
            async for _ in proj:
                pass


# ---------------------------------------------------------------------------
# ChatModelStream unit tests
# ---------------------------------------------------------------------------


class TestChatModelStream:
    """Test sync ChatModelStream with dispatch_event."""

    def test_text_projection_cached(self) -> None:
        stream = ChatModelStream()
        assert stream.text is stream.text

    def test_reasoning_projection_cached(self) -> None:
        stream = ChatModelStream()
        assert stream.reasoning is stream.reasoning

    def test_tool_calls_projection_cached(self) -> None:
        stream = ChatModelStream()
        assert stream.tool_calls is stream.tool_calls

    def test_text_deltas_via_pump(self) -> None:
        stream = ChatModelStream()
        events: list[MessagesData] = [
            {"event": "message-start", "role": "ai"},
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": "Hi"},
            },
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": " there"},
            },
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {"type": "text", "text": "Hi there"},
            },
            {"event": "message-finish", "reason": "stop"},
        ]
        idx = 0

        def pump() -> bool:
            nonlocal idx
            if idx >= len(events):
                return False
            dispatch_event(events[idx], stream)
            idx += 1
            return True

        stream._bind_pump(pump)
        assert list(stream.text) == ["Hi", " there"]
        assert str(stream.text) == "Hi there"

    def test_tool_call_chunk_streaming(self) -> None:
        stream = ChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {
                    "type": "tool_call_chunk",
                    "id": "tc1",
                    "name": "search",
                    "args": '{"q":',
                    "index": 0,
                },
            },
            stream,
        )
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {
                    "type": "tool_call_chunk",
                    "args": '"test"}',
                    "index": 0,
                },
            },
            stream,
        )
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "tool_call",
                    "id": "tc1",
                    "name": "search",
                    "args": {"q": "test"},
                },
            },
            stream,
        )
        dispatch_event({"event": "message-finish", "reason": "tool_use"}, stream)

        # Check chunk deltas were pushed
        chunks = list(stream.tool_calls)
        assert len(chunks) == 2  # two chunk deltas
        assert chunks[0]["type"] == "tool_call_chunk"
        assert chunks[0]["name"] == "search"

        # Check finalized tool calls
        finalized = stream.tool_calls.get()
        assert len(finalized) == 1
        assert finalized[0]["name"] == "search"
        assert finalized[0]["args"] == {"q": "test"}

    def test_multi_tool_parallel(self) -> None:
        stream = ChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        # Tool 1 starts
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {
                    "type": "tool_call_chunk",
                    "id": "t1",
                    "name": "foo",
                    "args": '{"a":',
                    "index": 0,
                },
            },
            stream,
        )
        # Tool 2 starts
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 1,
                "content_block": {
                    "type": "tool_call_chunk",
                    "id": "t2",
                    "name": "bar",
                    "args": '{"b":',
                    "index": 1,
                },
            },
            stream,
        )
        # Tool 1 finishes
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "tool_call",
                    "id": "t1",
                    "name": "foo",
                    "args": {"a": 1},
                },
            },
            stream,
        )
        # Tool 2 finishes
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 1,
                "content_block": {
                    "type": "tool_call",
                    "id": "t2",
                    "name": "bar",
                    "args": {"b": 2},
                },
            },
            stream,
        )
        dispatch_event({"event": "message-finish", "reason": "tool_use"}, stream)

        finalized = stream.tool_calls.get()
        assert len(finalized) == 2
        assert finalized[0]["name"] == "foo"
        assert finalized[1]["name"] == "bar"

    def test_output_assembles_aimessage(self) -> None:
        stream = ChatModelStream(message_id="msg-1")
        dispatch_event(
            {
                "event": "message-start",
                "role": "ai",
                "metadata": {"provider": "anthropic", "model": "claude-4"},
            },
            stream,
        )
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello"},
            },
            stream,
        )
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello"},
            },
            stream,
        )
        dispatch_event(
            {
                "event": "message-finish",
                "reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
            stream,
        )

        msg = stream.output
        assert msg.content == "Hello"
        assert msg.id == "msg-1"
        assert msg.response_metadata["finish_reason"] == "stop"
        assert msg.response_metadata["model_provider"] == "anthropic"
        assert msg.usage_metadata is not None
        assert msg.usage_metadata["input_tokens"] == 10

    def test_error_propagates_to_projections(self) -> None:
        stream = ChatModelStream()
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": "partial"},
            },
            stream,
        )
        stream._fail(RuntimeError("connection lost"))

        with pytest.raises(RuntimeError, match="connection lost"):
            str(stream.text)

        with pytest.raises(RuntimeError, match="connection lost"):
            stream.tool_calls.get()

    def test_raw_event_iteration(self) -> None:
        stream = ChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": "hi"},
            },
            stream,
        )
        dispatch_event({"event": "message-finish", "reason": "stop"}, stream)

        events = list(stream)
        assert len(events) == 3
        assert events[0]["event"] == "message-start"
        assert events[2]["event"] == "message-finish"

    def test_raw_event_multi_cursor(self) -> None:
        stream = ChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        dispatch_event({"event": "message-finish", "reason": "stop"}, stream)

        assert list(stream) == list(stream)  # Replay


# ---------------------------------------------------------------------------
# AsyncChatModelStream unit tests
# ---------------------------------------------------------------------------


class TestAsyncChatModelStream:
    """Test async ChatModelStream."""

    @pytest.mark.asyncio
    async def test_await_output(self) -> None:
        stream = AsyncChatModelStream(message_id="m1")

        async def produce() -> None:
            await asyncio.sleep(0)
            dispatch_event({"event": "message-start", "role": "ai"}, stream)
            dispatch_event(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "content_block": {"type": "text", "text": "Hi"},
                },
                stream,
            )
            dispatch_event({"event": "message-finish", "reason": "stop"}, stream)

        asyncio.get_running_loop().create_task(produce())
        msg = await stream
        assert msg.content == "Hi"

    @pytest.mark.asyncio
    async def test_async_text_deltas(self) -> None:
        stream = AsyncChatModelStream()

        async def produce() -> None:
            await asyncio.sleep(0)
            dispatch_event({"event": "message-start", "role": "ai"}, stream)
            await asyncio.sleep(0)
            dispatch_event(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "content_block": {"type": "text", "text": "a"},
                },
                stream,
            )
            await asyncio.sleep(0)
            dispatch_event(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "content_block": {"type": "text", "text": "b"},
                },
                stream,
            )
            await asyncio.sleep(0)
            dispatch_event({"event": "message-finish", "reason": "stop"}, stream)

        asyncio.get_running_loop().create_task(produce())
        deltas = [d async for d in stream.text]
        assert deltas == ["a", "b"]

    @pytest.mark.asyncio
    async def test_await_tool_calls(self) -> None:
        stream = AsyncChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {
                    "type": "tool_call_chunk",
                    "id": "tc1",
                    "name": "search",
                    "args": '{"q":"hi"}',
                    "index": 0,
                },
            },
            stream,
        )
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "tool_call",
                    "id": "tc1",
                    "name": "search",
                    "args": {"q": "hi"},
                },
            },
            stream,
        )
        dispatch_event({"event": "message-finish", "reason": "tool_use"}, stream)

        result = await stream.tool_calls
        assert len(result) == 1
        assert result[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_async_raw_event_iteration(self) -> None:
        stream = AsyncChatModelStream()

        async def produce() -> None:
            await asyncio.sleep(0)
            dispatch_event({"event": "message-start", "role": "ai"}, stream)
            await asyncio.sleep(0)
            dispatch_event({"event": "message-finish", "reason": "stop"}, stream)

        asyncio.get_running_loop().create_task(produce())
        events = [e async for e in stream]
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_error_propagation(self) -> None:
        stream = AsyncChatModelStream()
        stream._fail(RuntimeError("async fail"))

        with pytest.raises(RuntimeError, match="async fail"):
            await stream.text
        with pytest.raises(RuntimeError, match="async fail"):
            await stream
