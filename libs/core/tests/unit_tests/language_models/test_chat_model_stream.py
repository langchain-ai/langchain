"""Tests for ChatModelStream, AsyncChatModelStream, and projections."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest

from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    AsyncProjection,
    ChatModelStream,
    SyncProjection,
    SyncTextProjection,
)

if TYPE_CHECKING:
    from langchain_protocol.protocol import ContentBlockFinishData

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

    @pytest.mark.asyncio
    async def test_arequest_more_drives_iteration(self) -> None:
        """Cursor drives the async pump when the buffer is empty."""
        proj = AsyncProjection()
        deltas = iter(["a", "b", "c"])

        async def pump() -> bool:
            try:
                proj.push(next(deltas))
            except StopIteration:
                proj.complete("abc")
                return False
            return True

        proj.set_arequest_more(pump)
        collected = [d async for d in proj]
        assert collected == ["a", "b", "c"]
        assert await proj == "abc"

    @pytest.mark.asyncio
    async def test_arequest_more_drives_await(self) -> None:
        """`await projection` drives the pump too, not just iteration."""
        proj = AsyncProjection()
        steps = iter([("push", "x"), ("push", "y"), ("complete", "xy")])

        async def pump() -> bool:
            try:
                action, value = next(steps)
            except StopIteration:
                return False
            if action == "push":
                proj.push(value)
            else:
                proj.complete(value)
            return True

        proj.set_arequest_more(pump)
        assert await proj == "xy"

    @pytest.mark.asyncio
    async def test_arequest_more_stops_when_pump_exhausts(self) -> None:
        """Pump returning False without completing ends iteration cleanly."""
        proj = AsyncProjection()
        pushed = [False]

        async def pump() -> bool:
            if not pushed[0]:
                proj.push("only")
                pushed[0] = True
                return True
            return False

        proj.set_arequest_more(pump)
        collected = [d async for d in proj]
        assert collected == ["only"]

    @pytest.mark.asyncio
    async def test_async_chat_model_stream_set_arequest_more_fans_out(self) -> None:
        """`set_arequest_more` wires every projection on AsyncChatModelStream."""
        stream = AsyncChatModelStream(message_id="m1")

        async def pump() -> bool:
            return False

        stream.set_arequest_more(pump)
        for proj in (
            stream._text_proj,
            stream._reasoning_proj,
            stream._tool_calls_proj,
            stream._output_proj,
            stream._events_proj,
        ):
            assert proj._arequest_more is pump

    @pytest.mark.asyncio
    async def test_concurrent_text_and_output_share_pump(self) -> None:
        """Concurrent `stream.text` + `await stream.output` both drive the pump."""
        stream = AsyncChatModelStream(message_id="m1")

        events: list[dict[str, Any]] = [
            {
                "event": "message-start",
                "role": "ai",
                "message_id": "m1",
                "metadata": {"provider": "test", "model": "fake"},
            },
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": "hello "},
            },
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": "world"},
            },
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {"type": "text", "text": "hello world"},
            },
            {"event": "message-finish"},
        ]
        cursor = iter(events)
        pump_lock = asyncio.Lock()

        async def pump() -> bool:
            async with pump_lock:
                try:
                    evt = next(cursor)
                except StopIteration:
                    return False
                stream.dispatch(evt)
                return True

        stream.set_arequest_more(pump)

        async def drain_text() -> str:
            buf = [delta async for delta in stream.text]
            return "".join(buf)

        text, message = await asyncio.gather(drain_text(), stream.output)
        assert text == "hello world"
        assert message.content == [{"type": "text", "text": "hello world", "index": 0}]


# ---------------------------------------------------------------------------
# ChatModelStream unit tests
# ---------------------------------------------------------------------------


class TestChatModelStream:
    """Test sync ChatModelStream via `stream.dispatch`."""

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
        events: list[dict[str, Any]] = [
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
            {"event": "message-finish"},
        ]
        idx = 0

        def pump() -> bool:
            nonlocal idx
            if idx >= len(events):
                return False
            stream.dispatch(events[idx])
            idx += 1
            return True

        stream.bind_pump(pump)
        assert list(stream.text) == ["Hi", " there"]
        assert str(stream.text) == "Hi there"

    def test_tool_call_chunk_streaming(self) -> None:
        stream = ChatModelStream()
        stream.dispatch({"event": "message-start", "role": "ai"})
        stream.dispatch(
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
            }
        )
        stream.dispatch(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {
                    "type": "tool_call_chunk",
                    "args": '"test"}',
                    "index": 0,
                },
            }
        )
        stream.dispatch(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "tool_call",
                    "id": "tc1",
                    "name": "search",
                    "args": {"q": "test"},
                },
            }
        )
        stream.dispatch({"event": "message-finish"})

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
        stream.dispatch({"event": "message-start", "role": "ai"})
        # Tool 1 starts
        stream.dispatch(
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
            }
        )
        # Tool 2 starts
        stream.dispatch(
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
            }
        )
        # Tool 1 finishes
        stream.dispatch(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "tool_call",
                    "id": "t1",
                    "name": "foo",
                    "args": {"a": 1},
                },
            }
        )
        # Tool 2 finishes
        stream.dispatch(
            {
                "event": "content-block-finish",
                "index": 1,
                "content_block": {
                    "type": "tool_call",
                    "id": "t2",
                    "name": "bar",
                    "args": {"b": 2},
                },
            }
        )
        stream.dispatch({"event": "message-finish"})

        finalized = stream.tool_calls.get()
        assert len(finalized) == 2
        assert finalized[0]["name"] == "foo"
        assert finalized[1]["name"] == "bar"

    def test_output_assembles_aimessage(self) -> None:
        stream = ChatModelStream(message_id="msg-1")
        stream.dispatch(
            {
                "event": "message-start",
                "role": "ai",
                "metadata": {"provider": "anthropic", "model": "claude-4"},
            }
        )
        stream.dispatch(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello"},
            }
        )
        stream.dispatch(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello"},
            }
        )
        stream.dispatch(
            {
                "event": "message-finish",
                "metadata": {"finish_reason": "stop"},
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        )

        msg = stream.output
        assert msg.content == [{"type": "text", "text": "Hello", "index": 0}]
        assert msg.id == "msg-1"
        assert msg.response_metadata["finish_reason"] == "stop"
        assert msg.response_metadata["model_provider"] == "anthropic"
        assert msg.usage_metadata is not None
        assert msg.usage_metadata["input_tokens"] == 10

    def test_error_propagates_to_projections(self) -> None:
        stream = ChatModelStream()
        stream.dispatch(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": "partial"},
            }
        )
        stream.fail(RuntimeError("connection lost"))

        with pytest.raises(RuntimeError, match="connection lost"):
            str(stream.text)

        with pytest.raises(RuntimeError, match="connection lost"):
            stream.tool_calls.get()

    def test_raw_event_iteration(self) -> None:
        stream = ChatModelStream()
        stream.dispatch({"event": "message-start", "role": "ai"})
        stream.dispatch(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": "hi"},
            }
        )
        stream.dispatch({"event": "message-finish"})

        events = list(stream)
        assert len(events) == 3
        assert events[0]["event"] == "message-start"
        assert events[2]["event"] == "message-finish"

    def test_raw_event_multi_cursor(self) -> None:
        stream = ChatModelStream()
        stream.dispatch({"event": "message-start", "role": "ai"})
        stream.dispatch({"event": "message-finish"})

        assert list(stream) == list(stream)  # Replay

    def test_invalid_tool_call_preserved_on_finish(self) -> None:
        """An `invalid_tool_call` finish lands on `invalid_tool_calls`."""
        stream = ChatModelStream()
        stream.dispatch({"event": "message-start", "role": "ai"})
        stream.dispatch(
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
            }
        )
        stream.dispatch({"event": "message-finish"})

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
        stream.dispatch({"event": "message-start", "role": "ai"})
        # Stream a tool_call_chunk with malformed JSON args
        stream.dispatch(
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
            }
        )
        # Finish event declares the call invalid
        stream.dispatch(
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
            }
        )
        stream.dispatch({"event": "message-finish"})

        msg = stream.output
        # The sweep must NOT have revived the chunk as an empty-args tool_call.
        assert msg.tool_calls == []
        assert len(msg.invalid_tool_calls) == 1

    def test_output_content_uses_protocol_tool_call_shape(self) -> None:
        """`.output.content` must emit `type: tool_call`, not legacy tool_use."""
        stream = ChatModelStream()
        stream.dispatch({"event": "message-start", "role": "ai"})
        stream.dispatch(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {"type": "text", "text": "Let me search."},
            }
        )
        stream.dispatch(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {"type": "text", "text": "Let me search."},
            }
        )
        stream.dispatch(
            {
                "event": "content-block-finish",
                "index": 1,
                "content_block": {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "search",
                    "args": {"q": "weather"},
                },
            }
        )
        stream.dispatch({"event": "message-finish"})

        msg = stream.output
        assert isinstance(msg.content, list)
        content = cast("list[dict[str, Any]]", msg.content)
        types = [b.get("type") for b in content]
        assert types == ["text", "tool_call"]
        tool_block = content[1]
        assert tool_block["name"] == "search"
        assert tool_block["args"] == {"q": "weather"}
        # Legacy shape fields must be absent
        assert "input" not in tool_block
        assert tool_block.get("type") != "tool_use"

    def test_server_tool_call_finish_lands_in_output_content(self) -> None:
        """Server-executed tool call finish events flow into .output.content."""
        stream = ChatModelStream()
        stream.dispatch({"event": "message-start", "role": "ai"})
        stream.dispatch(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "server_tool_call",
                    "id": "srv_1",
                    "name": "web_search",
                    "args": {"q": "weather"},
                },
            }
        )
        stream.dispatch(
            cast(
                "ContentBlockFinishData",
                {
                    "event": "content-block-finish",
                    "index": 1,
                    "content_block": {
                        "type": "server_tool_result",
                        "tool_call_id": "srv_1",
                        "status": "success",
                        "output": "62F, clear",
                    },
                },
            )
        )
        stream.dispatch({"event": "message-finish"})

        msg = stream.output
        assert isinstance(msg.content, list)
        content = cast("list[dict[str, Any]]", msg.content)
        types = [b.get("type") for b in content]
        assert types == ["server_tool_call", "server_tool_result"]
        # Regular tool_calls projection must NOT include server-executed ones
        assert msg.tool_calls == []

    def test_server_tool_call_chunk_sweep(self) -> None:
        """Unfinished server_tool_call_chunks get swept to server_tool_call."""
        stream = ChatModelStream()
        stream.dispatch({"event": "message-start", "role": "ai"})
        stream.dispatch(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {
                    "type": "server_tool_call_chunk",
                    "id": "srv_1",
                    "name": "web_search",
                    "args": '{"q":',
                },
            }
        )
        stream.dispatch(
            {
                "event": "content-block-delta",
                "index": 0,
                "content_block": {
                    "type": "server_tool_call_chunk",
                    "args": ' "weather"}',
                },
            }
        )
        stream.dispatch({"event": "message-finish"})

        msg = stream.output
        assert isinstance(msg.content, list)
        content = cast("list[dict[str, Any]]", msg.content)
        assert content[0]["type"] == "server_tool_call"
        assert content[0]["args"] == {"q": "weather"}
        assert content[0]["name"] == "web_search"

    def test_image_block_pass_through(self) -> None:
        """An image block finished via the event stream reaches .output.content."""
        stream = ChatModelStream()
        stream.dispatch({"event": "message-start", "role": "ai"})
        stream.dispatch(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "image",
                    "url": "https://example.com/cat.png",
                    "mime_type": "image/png",
                },
            }
        )
        stream.dispatch({"event": "message-finish"})

        msg = stream.output
        assert isinstance(msg.content, list)
        assert msg.content[0] == {
            "type": "image",
            "url": "https://example.com/cat.png",
            "mime_type": "image/png",
            "index": 0,
        }

    def test_sweep_of_unfinished_malformed_chunk_produces_invalid_tool_call(
        self,
    ) -> None:
        """Unfinished chunk with malformed JSON sweeps to invalid_tool_call."""
        stream = ChatModelStream()
        stream.dispatch({"event": "message-start", "role": "ai"})
        stream.dispatch(
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
            }
        )
        stream.dispatch({"event": "message-finish"})

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
            stream.dispatch({"event": "message-start", "role": "ai"})
            stream.dispatch(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "content_block": {"type": "text", "text": "Hi"},
                }
            )
            stream.dispatch({"event": "message-finish"})

        asyncio.get_running_loop().create_task(produce())
        msg = await stream
        assert msg.content == "Hi"

    @pytest.mark.asyncio
    async def test_async_text_deltas(self) -> None:
        stream = AsyncChatModelStream()

        async def produce() -> None:
            await asyncio.sleep(0)
            stream.dispatch({"event": "message-start", "role": "ai"})
            await asyncio.sleep(0)
            stream.dispatch(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "content_block": {"type": "text", "text": "a"},
                }
            )
            await asyncio.sleep(0)
            stream.dispatch(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "content_block": {"type": "text", "text": "b"},
                }
            )
            await asyncio.sleep(0)
            stream.dispatch({"event": "message-finish"})

        asyncio.get_running_loop().create_task(produce())
        deltas = [d async for d in stream.text]
        assert deltas == ["a", "b"]

    @pytest.mark.asyncio
    async def test_await_tool_calls(self) -> None:
        stream = AsyncChatModelStream()
        stream.dispatch({"event": "message-start", "role": "ai"})
        stream.dispatch(
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
            }
        )
        stream.dispatch(
            {
                "event": "content-block-finish",
                "index": 0,
                "content_block": {
                    "type": "tool_call",
                    "id": "tc1",
                    "name": "search",
                    "args": {"q": "hi"},
                },
            }
        )
        stream.dispatch({"event": "message-finish"})

        result = await stream.tool_calls
        assert len(result) == 1
        assert result[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_async_raw_event_iteration(self) -> None:
        stream = AsyncChatModelStream()

        async def produce() -> None:
            await asyncio.sleep(0)
            stream.dispatch({"event": "message-start", "role": "ai"})
            await asyncio.sleep(0)
            stream.dispatch({"event": "message-finish"})

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
