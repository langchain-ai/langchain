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
        proj.push("a")
        proj.push("b")
        proj.complete(["a", "b"])
        assert list(proj) == ["a", "b"]

    def test_get_returns_final_value(self) -> None:
        proj = SyncProjection()
        proj.push("x")
        proj.complete("final")
        assert proj.get() == "final"

    def test_request_more_pulls(self) -> None:
        proj = SyncProjection()
        calls = iter(["a", "b", None])

        def pump() -> bool:
            val = next(calls)
            if val is None:
                proj.complete("ab")
                return True
            proj.push(val)
            return True

        proj._request_more = pump
        assert list(proj) == ["a", "b"]
        assert proj.get() == "ab"

    def test_error_propagation(self) -> None:
        proj = SyncProjection()
        proj.push("partial")
        proj.fail(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            list(proj)

    def test_error_on_get(self) -> None:
        proj = SyncProjection()
        proj.fail(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            proj.get()

    def test_multi_cursor_replay(self) -> None:
        proj = SyncProjection()
        proj.push("a")
        proj.push("b")
        proj.complete(None)
        assert list(proj) == ["a", "b"]
        assert list(proj) == ["a", "b"]  # Second iteration replays

    def test_empty_projection(self) -> None:
        proj = SyncProjection()
        proj.complete([])
        assert list(proj) == []
        assert proj.get() == []


class TestSyncTextProjection:
    """Test SyncTextProjection string convenience methods."""

    def test_str_drains(self) -> None:
        proj = SyncTextProjection()
        proj.push("Hello")
        proj.push(" world")
        proj.complete("Hello world")
        assert str(proj) == "Hello world"

    def test_str_with_pump(self) -> None:
        proj = SyncTextProjection()
        done = False

        def pump() -> bool:
            nonlocal done
            if not done:
                proj.push("Hi")
                proj.complete("Hi")
                done = True
                return True
            return False

        proj._request_more = pump
        assert str(proj) == "Hi"

    def test_bool_nonempty(self) -> None:
        proj = SyncTextProjection()
        assert not proj
        proj.push("x")
        assert proj

    def test_repr(self) -> None:
        proj = SyncTextProjection()
        proj.push("hello")
        assert repr(proj) == "'hello'"
        proj.complete("hello")
        assert repr(proj) == "'hello'"


class TestAsyncProjection:
    """Test AsyncProjection async iteration and awaiting."""

    @pytest.mark.asyncio
    async def test_await_final_value(self) -> None:
        proj = AsyncProjection()
        proj.push("a")
        proj.complete("final")
        assert await proj == "final"

    @pytest.mark.asyncio
    async def test_async_iter(self) -> None:
        proj = AsyncProjection()

        async def produce() -> None:
            await asyncio.sleep(0)
            proj.push("x")
            await asyncio.sleep(0)
            proj.push("y")
            await asyncio.sleep(0)
            proj.complete("xy")

        asyncio.get_running_loop().create_task(produce())
        deltas = [d async for d in proj]
        assert deltas == ["x", "y"]

    @pytest.mark.asyncio
    async def test_error_on_await(self) -> None:
        proj = AsyncProjection()
        proj.fail(ValueError("async boom"))
        with pytest.raises(ValueError, match="async boom"):
            await proj

    @pytest.mark.asyncio
    async def test_error_on_iter(self) -> None:
        proj = AsyncProjection()
        proj.push("partial")
        proj.fail(ValueError("mid-stream"))
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

        stream.bind_pump(pump)
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
        stream.fail(RuntimeError("connection lost"))

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

    def test_invalid_tool_call_preserved_on_finish(self) -> None:
        """An ``invalid_tool_call`` finish lands on ``invalid_tool_calls``."""
        stream = ChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "invalid_tool_call",
                    "id": "call_1",
                    "name": "search",
                    "args": '{"q": ',  # malformed
                    "error": "Failed to parse tool call arguments as JSON",
                },
            },
            stream,
        )
        dispatch_event({"event": "message-finish", "reason": "stop"}, stream)

        msg = stream.output
        assert msg.tool_calls == []
        assert len(msg.invalid_tool_calls) == 1
        assert msg.invalid_tool_calls[0]["name"] == "search"
        assert msg.invalid_tool_calls[0]["args"] == '{"q": '
        assert msg.invalid_tool_calls[0]["error"] == (
            "Failed to parse tool call arguments as JSON"
        )

    def test_invalid_tool_call_survives_sweep(self) -> None:
        """Regression: finish deletes stale chunk, sweep cannot revive it."""
        stream = ChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        # Stream a tool_call_chunk with malformed JSON args
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {
                    "type": "tool_call_chunk",
                    "id": "call_1",
                    "name": "search",
                    "args": '{"q": ',
                    "index": 0,
                },
            },
            stream,
        )
        # Finish event declares the call invalid
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "invalid_tool_call",
                    "id": "call_1",
                    "name": "search",
                    "args": '{"q": ',
                    "error": "Failed to parse tool call arguments as JSON",
                },
            },
            stream,
        )
        dispatch_event({"event": "message-finish", "reason": "stop"}, stream)

        msg = stream.output
        # The sweep must NOT have revived the chunk as an empty-args tool_call.
        assert msg.tool_calls == []
        assert len(msg.invalid_tool_calls) == 1

    def test_output_content_uses_protocol_tool_call_shape(self) -> None:
        """`.output.content` must emit `type: tool_call`, not legacy tool_use."""
        stream = ChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": "Let me search."},
            },
            stream,
        )
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {"type": "text", "text": "Let me search."},
            },
            stream,
        )
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 1,
                "content_block": {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "search",
                    "args": {"q": "weather"},
                },
            },
            stream,
        )
        dispatch_event({"event": "message-finish", "reason": "tool_use"}, stream)

        msg = stream.output
        assert isinstance(msg.content, list)
        types = [b.get("type") for b in msg.content]
        assert types == ["text", "tool_call"]
        tool_block = msg.content[1]
        assert tool_block["name"] == "search"
        assert tool_block["args"] == {"q": "weather"}
        # Legacy shape fields must be absent
        assert "input" not in tool_block
        assert tool_block.get("type") != "tool_use"

    def test_server_tool_call_finish_lands_in_output_content(self) -> None:
        """Server-executed tool call finish events flow into .output.content."""
        stream = ChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "server_tool_call",
                    "id": "srv_1",
                    "name": "web_search",
                    "args": {"q": "weather"},
                },
            },
            stream,
        )
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 1,
                "content_block": {
                    "type": "server_tool_call_result",
                    "tool_call_id": "srv_1",
                    "status": "success",
                    "output": "62F, clear",
                },
            },
            stream,
        )
        dispatch_event({"event": "message-finish", "reason": "stop"}, stream)

        msg = stream.output
        assert isinstance(msg.content, list)
        types = [b.get("type") for b in msg.content]
        assert types == ["server_tool_call", "server_tool_call_result"]
        # Regular tool_calls projection must NOT include server-executed ones
        assert msg.tool_calls == []

    def test_server_tool_call_chunk_sweep(self) -> None:
        """Unfinished server_tool_call_chunks get swept to server_tool_call."""
        stream = ChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {
                    "type": "server_tool_call_chunk",
                    "id": "srv_1",
                    "name": "web_search",
                    "args": '{"q":',
                },
            },
            stream,
        )
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {
                    "type": "server_tool_call_chunk",
                    "args": ' "weather"}',
                },
            },
            stream,
        )
        dispatch_event({"event": "message-finish", "reason": "stop"}, stream)

        msg = stream.output
        assert isinstance(msg.content, list)
        assert msg.content[0]["type"] == "server_tool_call"
        assert msg.content[0]["args"] == {"q": "weather"}
        assert msg.content[0]["name"] == "web_search"

    def test_image_block_pass_through(self) -> None:
        """An image block finished via the event stream reaches .output.content."""
        stream = ChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        dispatch_event(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "image",
                    "url": "https://example.com/cat.png",
                    "mime_type": "image/png",
                },
            },
            stream,
        )
        dispatch_event({"event": "message-finish", "reason": "stop"}, stream)

        msg = stream.output
        assert isinstance(msg.content, list)
        assert msg.content[0] == {
            "type": "image",
            "url": "https://example.com/cat.png",
            "mime_type": "image/png",
        }

    def test_sweep_of_unfinished_malformed_chunk_produces_invalid_tool_call(
        self,
    ) -> None:
        """Unfinished chunk with malformed JSON sweeps to invalid_tool_call."""
        stream = ChatModelStream()
        dispatch_event({"event": "message-start", "role": "ai"}, stream)
        dispatch_event(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {
                    "type": "tool_call_chunk",
                    "id": "call_1",
                    "name": "search",
                    "args": '{"q": ',  # malformed, never completed
                    "index": 0,
                },
            },
            stream,
        )
        dispatch_event({"event": "message-finish", "reason": "stop"}, stream)

        msg = stream.output
        assert msg.tool_calls == []
        assert len(msg.invalid_tool_calls) == 1
        itc = msg.invalid_tool_calls[0]
        assert itc["name"] == "search"
        assert itc["args"] == '{"q": '
        assert "Failed to parse" in (itc["error"] or "")


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
        stream.fail(RuntimeError("async fail"))

        with pytest.raises(RuntimeError, match="async fail"):
            await stream.text
        with pytest.raises(RuntimeError, match="async fail"):
            await stream
