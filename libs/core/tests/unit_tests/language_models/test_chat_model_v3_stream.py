"""Tests for `stream_events(version="v3")` (sync + async) and `ChatModelStream`."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest
from langchain_protocol.protocol import (
    ContentBlockDeltaData,
    ContentBlockFinishData,
    MessageFinishData,
    ReasoningContentBlock,
    TextContentBlock,
    ToolCall,
    UsageInfo,
)

from langchain_core.caches import InMemoryCache
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    ChatModelStream,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.tracers._streaming import _V2StreamingCallbackHandler

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage


class _MalformedToolCallModel(BaseChatModel):
    """Fake model that emits a tool_call_chunk with malformed JSON args."""

    @property
    def _llm_type(self) -> str:
        return "malformed-tool-call-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise NotImplementedError

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "search",
                        "args": '{"q": ',  # malformed JSON
                        "id": "call_1",
                        "index": 0,
                    }
                ],
            )
        )


class _AnthropicStyleServerToolModel(BaseChatModel):
    """Fake model that streams Anthropic-native server_tool_use shapes.

    Exercises Phase E: the bridge should call `content_blocks` (which
    invokes the Anthropic translator) to convert `server_tool_use` into
    protocol `server_tool_call` blocks instead of silently dropping them.
    """

    @property
    def _llm_type(self) -> str:
        return "anthropic-style-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise NotImplementedError

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        # Single chunk carrying a complete server_tool_use block — what
        # Anthropic typically emits once input_json_delta finishes.
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_01",
                        "name": "web_search",
                        "input": {"query": "weather today"},
                    },
                    {"type": "text", "text": "Based on the search..."},
                ],
                response_metadata={"model_provider": "anthropic"},
            )
        )


class TestChatModelStream:
    """Test the sync ChatModelStream object."""

    def test_push_text_delta(self) -> None:
        stream = ChatModelStream()
        stream._push_content_block_delta(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "text-delta", "text": "Hello"},
            )
        )
        assert stream._text_acc == "Hello"

    def test_push_reasoning_delta(self) -> None:
        stream = ChatModelStream()
        stream._push_content_block_delta(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "reasoning-delta", "reasoning": "think"},
            )
        )
        assert stream._reasoning_acc == "think"

    def test_push_content_block_finish_tool_call(self) -> None:
        stream = ChatModelStream()
        stream._push_content_block_finish(
            ContentBlockFinishData(
                event="content-block-finish",
                index=0,
                content=ToolCall(
                    type="tool_call",
                    id="tc1",
                    name="search",
                    args={"q": "test"},
                ),
            )
        )
        assert len(stream._tool_calls_acc) == 1
        assert stream._tool_calls_acc[0]["name"] == "search"

    def test_finish(self) -> None:
        stream = ChatModelStream()
        assert not stream.done
        usage = UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15)
        stream._finish(MessageFinishData(event="message-finish", usage=usage))
        assert stream.done
        assert stream._usage_value == usage

    def test_fail(self) -> None:
        stream = ChatModelStream()
        stream.fail(RuntimeError("test"))
        assert stream.done

    def test_pump_driven_text(self) -> None:
        """Test text projection with pump binding."""
        stream = ChatModelStream()
        deltas: list[ContentBlockDeltaData] = [
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "text-delta", "text": "Hi"},
            ),
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "text-delta", "text": " there"},
            ),
        ]
        finish = MessageFinishData(event="message-finish")
        idx = 0

        def pump_one() -> bool:
            nonlocal idx
            if idx < len(deltas):
                stream._push_content_block_delta(deltas[idx])
                idx += 1
                return True
            if idx == len(deltas):
                stream._finish(finish)
                idx += 1
                return True
            return False

        stream.bind_pump(pump_one)

        text_deltas = list(stream.text)
        assert text_deltas == ["Hi", " there"]
        assert stream.done


class TestAsyncChatModelStream:
    """Test the async ChatModelStream object."""

    @pytest.mark.asyncio
    async def test_text_await(self) -> None:
        stream = AsyncChatModelStream()
        stream._push_content_block_delta(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "text-delta", "text": "Hello"},
            )
        )
        stream._push_content_block_delta(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "text-delta", "text": " world"},
            )
        )
        stream._finish(MessageFinishData(event="message-finish"))

        full = await stream.text
        assert full == "Hello world"

    @pytest.mark.asyncio
    async def test_text_async_iter(self) -> None:
        stream = AsyncChatModelStream()

        async def produce() -> None:
            await asyncio.sleep(0)
            stream._push_content_block_delta(
                ContentBlockDeltaData(
                    event="content-block-delta",
                    index=0,
                    delta={"type": "text-delta", "text": "a"},
                )
            )
            await asyncio.sleep(0)
            stream._push_content_block_delta(
                ContentBlockDeltaData(
                    event="content-block-delta",
                    index=0,
                    delta={"type": "text-delta", "text": "b"},
                )
            )
            await asyncio.sleep(0)
            stream._finish(MessageFinishData(event="message-finish"))

        asyncio.get_running_loop().create_task(produce())

        deltas = [d async for d in stream.text]
        assert deltas == ["a", "b"]

    @pytest.mark.asyncio
    async def test_tool_calls_await(self) -> None:
        stream = AsyncChatModelStream()
        stream._push_content_block_finish(
            ContentBlockFinishData(
                event="content-block-finish",
                index=0,
                content=ToolCall(
                    type="tool_call",
                    id="tc1",
                    name="search",
                    args={"q": "test"},
                ),
            )
        )
        stream._finish(MessageFinishData(event="message-finish"))

        tool_calls = await stream.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_error_propagation(self) -> None:
        stream = AsyncChatModelStream()
        stream.fail(RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await stream.text


class TestStreamV2:
    """Test `BaseChatModel.stream_events(version="v3")` with `FakeListChatModel`."""

    def test_stream_events_v3_text(self) -> None:
        model = FakeListChatModel(responses=["Hello world!"])
        stream = model.stream_events("test", version="v3")

        assert isinstance(stream, ChatModelStream)
        deltas = list(stream.text)
        assert "".join(deltas) == "Hello world!"
        assert stream.done

    def test_stream_events_v3_usage(self) -> None:
        model = FakeListChatModel(responses=["Hi"])
        stream = model.stream_events("test", version="v3")

        # Drain stream
        for _ in stream.text:
            pass
        # FakeListChatModel doesn't emit usage, so it should be None
        assert stream.output.usage_metadata is None
        assert stream.done

    def test_stream_events_v3_malformed_tool_args_produce_invalid_tool_call(
        self,
    ) -> None:
        """End-to-end: malformed tool-call JSON becomes invalid_tool_calls."""
        model = _MalformedToolCallModel()
        stream = model.stream_events("test", version="v3")
        msg = stream.output

        assert msg.tool_calls == []
        assert len(msg.invalid_tool_calls) == 1
        itc = msg.invalid_tool_calls[0]
        assert itc["name"] == "search"
        assert itc["args"] == '{"q": '
        assert itc["id"] == "call_1"

    def test_stream_events_v3_translates_anthropic_server_tool_use_to_protocol(
        self,
    ) -> None:
        """Phase E end-to-end: server_tool_use becomes server_tool_call in output."""
        model = _AnthropicStyleServerToolModel()
        stream = model.stream_events("weather?", version="v3")
        msg = stream.output

        assert isinstance(msg.content, list)
        types = [b.get("type") for b in msg.content if isinstance(b, dict)]
        # The server tool call must appear in the output content.
        assert "server_tool_call" in types
        # Text block should also be present.
        assert "text" in types
        # Regular tool_calls should NOT include the server-executed call.
        assert msg.tool_calls == []


class TestAstreamV2:
    """Test `BaseChatModel.astream_events(version="v3")` with `FakeListChatModel`."""

    @pytest.mark.asyncio
    async def test_astream_events_v3_text(self) -> None:
        model = FakeListChatModel(responses=["Hello!"])
        stream = await model.astream_events("test", version="v3")

        assert isinstance(stream, AsyncChatModelStream)
        full = await stream.text
        assert full == "Hello!"

    @pytest.mark.asyncio
    async def test_astream_events_v3_deltas(self) -> None:
        model = FakeListChatModel(responses=["Hi"])
        stream = await model.astream_events("test", version="v3")

        deltas = [d async for d in stream.text]
        assert "".join(deltas) == "Hi"


class TestPerBlockAccumulation:
    """Regression: per-block text/reasoning must not cross-contaminate.

    When a message contains more than one `text` or `reasoning` block
    (Anthropic interleaves text around `tool_use`; OpenAI Responses
    emits multiple reasoning summary items), each finalized block must
    carry only its own payload — not the running message-wide total.
    """

    def test_two_text_blocks_keep_their_own_text(self) -> None:
        stream = ChatModelStream()
        # Block 0: "A"
        stream.dispatch(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "text-delta", "text": "A"},
            )
        )
        stream.dispatch(
            ContentBlockFinishData(
                event="content-block-finish",
                index=0,
                content=TextContentBlock(type="text", text="A"),
            )
        )
        # Block 1: "B"
        stream.dispatch(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=1,
                delta={"type": "text-delta", "text": "B"},
            )
        )
        stream.dispatch(
            ContentBlockFinishData(
                event="content-block-finish",
                index=1,
                content=TextContentBlock(type="text", text="B"),
            )
        )
        stream.dispatch(MessageFinishData(event="message-finish"))

        content = stream.output.content
        assert isinstance(content, list)
        text_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert [b["text"] for b in text_blocks] == ["A", "B"], (
            "Finalized text blocks must carry their own payloads, not the "
            "concatenation of all earlier text blocks."
        )
        # Message-wide projection still sums to the full text.
        assert str(stream.text) == "AB"

    def test_two_reasoning_blocks_keep_their_own_text(self) -> None:
        stream = ChatModelStream()
        # Block 0: "one"
        stream.dispatch(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "reasoning-delta", "reasoning": "one"},
            )
        )
        stream.dispatch(
            ContentBlockFinishData(
                event="content-block-finish",
                index=0,
                content=ReasoningContentBlock(type="reasoning", reasoning="one"),
            )
        )
        # Block 1: "two"
        stream.dispatch(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=1,
                delta={"type": "reasoning-delta", "reasoning": "two"},
            )
        )
        stream.dispatch(
            ContentBlockFinishData(
                event="content-block-finish",
                index=1,
                content=ReasoningContentBlock(type="reasoning", reasoning="two"),
            )
        )
        stream.dispatch(MessageFinishData(event="message-finish"))

        content = stream.output.content
        assert isinstance(content, list)
        reasoning_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "reasoning"
        ]
        assert [b["reasoning"] for b in reasoning_blocks] == ["one", "two"]
        assert str(stream.reasoning) == "onetwo"

    def test_finish_text_reconciles_with_partial_deltas(self) -> None:
        """`.text` must agree with `.output.content` when finish corrects deltas.

        If deltas stream "hel" and the `content-block-finish` payload
        carries the authoritative "hello", both the per-block finalized
        text and the message-wide projection must land on "hello".
        """
        stream = ChatModelStream()
        stream.dispatch(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "text-delta", "text": "hel"},
            )
        )
        stream.dispatch(
            ContentBlockFinishData(
                event="content-block-finish",
                index=0,
                content=TextContentBlock(type="text", text="hello"),
            )
        )
        stream.dispatch(MessageFinishData(event="message-finish"))

        content = stream.output.content
        assert isinstance(content, list)
        text_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert [b["text"] for b in text_blocks] == ["hello"]
        assert str(stream.text) == "hello"

    def test_out_of_order_finish_still_produces_correct_final_text(self) -> None:
        """Reconciliation must not depend on `_text_acc` suffix layout.

        If block 0 finishes with authoritative text *after* block 1 has
        already emitted deltas (possible in theory for a native
        `_stream_chat_model_events` provider, or any future mutation
        path that touches `_text_acc`), the in-place splice would
        corrupt the message-wide accumulator. The final value must be
        derived from per-block storage so both `stream.output.content`
        and `str(stream.text)` remain correct regardless of finish
        ordering.
        """
        stream = ChatModelStream()
        # Block 0 streams deltas first.
        stream.dispatch(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "text-delta", "text": "aaa"},
            )
        )
        # Block 1 streams deltas before block 0 finishes.
        stream.dispatch(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=1,
                delta={"type": "text-delta", "text": "bb"},
            )
        )
        # Block 0 finishes with authoritative text different from deltas.
        stream.dispatch(
            ContentBlockFinishData(
                event="content-block-finish",
                index=0,
                content=TextContentBlock(type="text", text="XXX"),
            )
        )
        stream.dispatch(
            ContentBlockFinishData(
                event="content-block-finish",
                index=1,
                content=TextContentBlock(type="text", text="bb"),
            )
        )
        stream.dispatch(MessageFinishData(event="message-finish"))

        content = stream.output.content
        assert isinstance(content, list)
        text_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert [b["text"] for b in text_blocks] == ["XXX", "bb"]
        # `str(stream.text)` must reflect the authoritative per-block
        # concatenation, not the splice-in-place result ("aaXXX") that
        # would have been left over from the old suffix assumption.
        assert str(stream.text) == "XXXbb"

    def test_finish_reasoning_reconciles_with_partial_deltas(self) -> None:
        """Same reconciliation invariant for the reasoning projection."""
        stream = ChatModelStream()
        stream.dispatch(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "reasoning-delta", "reasoning": "thi"},
            )
        )
        stream.dispatch(
            ContentBlockFinishData(
                event="content-block-finish",
                index=0,
                content=ReasoningContentBlock(type="reasoning", reasoning="thinking"),
            )
        )
        stream.dispatch(MessageFinishData(event="message-finish"))

        content = stream.output.content
        assert isinstance(content, list)
        reasoning_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "reasoning"
        ]
        assert [b["reasoning"] for b in reasoning_blocks] == ["thinking"]
        assert str(stream.reasoning) == "thinking"

    def test_interleaved_text_blocks_around_tool_call(self) -> None:
        """Anthropic shape: text, then tool_call, then more text."""
        stream = ChatModelStream()
        # Block 0: text "before"
        stream.dispatch(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                delta={"type": "text-delta", "text": "before"},
            )
        )
        stream.dispatch(
            ContentBlockFinishData(
                event="content-block-finish",
                index=0,
                content=TextContentBlock(type="text", text="before"),
            )
        )
        # Block 1: tool_call
        stream.dispatch(
            ContentBlockFinishData(
                event="content-block-finish",
                index=1,
                content=ToolCall(
                    type="tool_call",
                    id="tc1",
                    name="search",
                    args={"q": "x"},
                ),
            )
        )
        # Block 2: text "after"
        stream.dispatch(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=2,
                delta={"type": "text-delta", "text": "after"},
            )
        )
        stream.dispatch(
            ContentBlockFinishData(
                event="content-block-finish",
                index=2,
                content=TextContentBlock(type="text", text="after"),
            )
        )
        stream.dispatch(MessageFinishData(event="message-finish"))

        content = stream.output.content
        assert isinstance(content, list)
        text_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert [b["text"] for b in text_blocks] == ["before", "after"]


class _RecordingStreamModel(BaseChatModel):
    """Fake model that records the kwargs passed to _stream / _astream."""

    last_stream_kwargs: dict[str, Any] = {}  # noqa: RUF012
    last_astream_kwargs: dict[str, Any] = {}  # noqa: RUF012

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise NotImplementedError

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, run_manager
        type(self).last_stream_kwargs = dict(kwargs)
        yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        del messages, stop, run_manager
        type(self).last_astream_kwargs = dict(kwargs)
        yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))


class TestStructuredOutputKwargStripping:
    """Regression: structured-output tracing kwargs must not reach _stream.

    `stream()` / `astream()` pop `ls_structured_output_format` and
    `structured_output_format` before forwarding kwargs to `_stream` —
    provider clients reject unknown kwargs. `stream_events(version="v3")` /
    `astream_events(version="v3")` must do the same, or
    `.with_structured_output().stream_events(version="v3")` breaks.
    """

    def test_stream_events_v3_strips_ls_structured_output_format(self) -> None:
        model = _RecordingStreamModel()
        bound = model.bind(ls_structured_output_format={"schema": {"type": "object"}})
        stream = bound.stream_events("test", version="v3")
        _ = stream.output
        recorded = _RecordingStreamModel.last_stream_kwargs
        assert "ls_structured_output_format" not in recorded
        assert "structured_output_format" not in recorded

    def test_stream_events_v3_strips_structured_output_format(self) -> None:
        model = _RecordingStreamModel()
        bound = model.bind(structured_output_format={"schema": {"type": "object"}})
        stream = bound.stream_events("test", version="v3")
        _ = stream.output
        recorded = _RecordingStreamModel.last_stream_kwargs
        assert "ls_structured_output_format" not in recorded
        assert "structured_output_format" not in recorded

    @pytest.mark.asyncio
    async def test_astream_events_v3_strips_ls_structured_output_format(self) -> None:
        model = _RecordingStreamModel()
        bound = model.bind(ls_structured_output_format={"schema": {"type": "object"}})
        stream = await bound.astream_events("test", version="v3")
        _ = await stream
        assert (
            "ls_structured_output_format"
            not in _RecordingStreamModel.last_astream_kwargs
        )
        assert (
            "structured_output_format" not in _RecordingStreamModel.last_astream_kwargs
        )

    @pytest.mark.asyncio
    async def test_astream_events_v3_strips_structured_output_format(self) -> None:
        model = _RecordingStreamModel()
        bound = model.bind(structured_output_format={"schema": {"type": "object"}})
        stream = await bound.astream_events("test", version="v3")
        _ = await stream
        assert (
            "ls_structured_output_format"
            not in _RecordingStreamModel.last_astream_kwargs
        )
        assert (
            "structured_output_format" not in _RecordingStreamModel.last_astream_kwargs
        )


class _SlowTeardownModel(BaseChatModel):
    """Fake model whose `_astream` blocks cancellation teardown on a gate.

    Used to exercise the caller-cancellation path in `aclose()`:
    cancelling the producer causes it to enter a `CancelledError`
    handler that waits on `teardown_gate` before re-raising. That
    keeps the producer task in a "cancelled-but-not-done" state long
    enough for the test to cancel `aclose`'s caller deterministically.
    """

    def __init__(self, teardown_gate: asyncio.Event, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._teardown_gate = teardown_gate

    @property
    def _llm_type(self) -> str:
        return "slow-teardown-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise NotImplementedError

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        del messages, stop, run_manager, kwargs
        yield ChatGenerationChunk(message=AIMessageChunk(content="first"))
        # Block forever; cancellation is the only way out.
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Hold the cancellation teardown open until the test releases
            # the gate. The task stays in a pending state while this
            # handler is suspended, so `await task` on the `aclose()`
            # side remains blocked.
            await self._teardown_gate.wait()
            raise


class _GatedStreamModel(BaseChatModel):
    """Fake model whose _astream blocks on an event until released.

    Used to exercise consumer-cancellation cleanup: the producer task
    is parked inside `_astream` awaiting the gate, and `aclose()` must
    cancel it rather than leave it running.
    """

    def __init__(self, gate: asyncio.Event, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._gate = gate
        self._cancelled = False

    @property
    def _llm_type(self) -> str:
        return "gated-fake"

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise NotImplementedError

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        del messages, stop, run_manager, kwargs
        yield ChatGenerationChunk(message=AIMessageChunk(content="first"))
        try:
            await self._gate.wait()
        except asyncio.CancelledError:
            self._cancelled = True
            raise
        yield ChatGenerationChunk(message=AIMessageChunk(content="second"))


class TestAsyncStreamAclose:
    """Regression: aclose() must cancel the background producer task."""

    @pytest.mark.asyncio
    async def test_aclose_cancels_producer_task(self) -> None:
        gate = asyncio.Event()
        model = _GatedStreamModel(gate=gate)
        stream = await model.astream_events("test", version="v3")

        # Pull the first delta so the producer enters the gated section.
        aiter_ = stream.text.__aiter__()
        first = await aiter_.__anext__()
        assert first == "first"
        assert stream._producer_task is not None
        assert not stream._producer_task.done()

        await stream.aclose()

        assert stream._producer_task.done()
        assert stream._producer_task.cancelled() or model.cancelled

    @pytest.mark.asyncio
    async def test_aclose_is_idempotent(self) -> None:
        gate = asyncio.Event()
        model = _GatedStreamModel(gate=gate)
        stream = await model.astream_events("test", version="v3")
        aiter_ = stream.text.__aiter__()
        await aiter_.__anext__()

        await stream.aclose()
        await stream.aclose()  # second call must not raise

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_stream(self) -> None:
        gate = asyncio.Event()
        model = _GatedStreamModel(gate=gate)
        stream = await model.astream_events("test", version="v3")

        async with stream as s:
            assert s is stream
            aiter_ = stream.text.__aiter__()
            await aiter_.__anext__()

        assert stream._producer_task is not None
        assert stream._producer_task.done()

    @pytest.mark.asyncio
    async def test_aclose_propagates_caller_cancellation(self) -> None:
        """`aclose()` must not swallow cancellation of its caller.

        Uses `_SlowTeardownModel`, whose cancelled producer blocks
        inside its `CancelledError` handler waiting on `teardown_gate`.
        That keeps the producer task pending long enough for the test
        to cancel the closer task while it is genuinely suspended
        inside `aclose()` — exercising the caller-cancel propagation
        path deterministically on all Python versions.
        """
        teardown_gate = asyncio.Event()
        model = _SlowTeardownModel(teardown_gate=teardown_gate)
        stream = await model.astream_events("test", version="v3")

        # Prime the producer so it enters `_astream`'s forever-blocking
        # await.
        aiter_ = stream.text.__aiter__()
        await aiter_.__anext__()

        closer_returned_normally = False

        async def closer() -> None:
            nonlocal closer_returned_normally
            await stream.aclose()
            closer_returned_normally = True

        closer_task = asyncio.create_task(closer())
        # Pump the loop until the producer has been cancelled and has
        # entered its cancellation-teardown suspension on
        # `teardown_gate`. At that point `closer` is guaranteed to be
        # suspended inside `aclose`'s linked-future await.
        for _ in range(10):
            await asyncio.sleep(0)
            assert stream._producer_task is not None
            if (
                stream._producer_task is not None
                and not stream._producer_task.done()
                and not closer_task.done()
            ):
                break

        assert stream._producer_task is not None
        assert not stream._producer_task.done(), (
            "producer task should be parked in its cancellation teardown"
        )
        assert not closer_task.done(), "closer must still be inside aclose"

        closer_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closer_task
        assert not closer_returned_normally

        # Release the producer so it can finish cancellation, then
        # await it to avoid leaking a pending task out of the test.
        teardown_gate.set()
        with contextlib.suppress(BaseException):
            await stream._producer_task

    @pytest.mark.asyncio
    async def test_aclose_before_producer_starts_resolves_projections(self) -> None:
        """Early-cancel path: `_produce` never runs.

        If a consumer calls `astream_events(version="v3")` and immediately `aclose()`
        (or `async with` exits before the loop schedules `_produce`),
        `task.cancel()` marks the task cancelled without ever invoking
        its body — so neither `stream.fail` nor `on_llm_error` fires.
        Consumers awaiting `stream.output` / `stream.text` would hang
        forever without explicit cleanup in `aclose()`.
        """
        error_events: list[BaseException] = []

        class RecordingHandler(AsyncCallbackHandler):
            async def on_llm_error(self, error: BaseException, **_: Any) -> None:
                error_events.append(error)

        handler = RecordingHandler()
        gate = asyncio.Event()
        model = _GatedStreamModel(gate=gate)
        stream = await model.astream_events(
            "test", config={"callbacks": [handler]}, version="v3"
        )
        # No yield to the event loop between `astream_events(version="v3")`
        # returning and `aclose()` — the producer task has been created
        # but its body has not executed.
        await stream.aclose()

        # `await stream.output` must resolve (with CancelledError)
        # rather than hang.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(stream.output, timeout=1.0)

        # `on_llm_error` must have been invoked for tracing continuity,
        # even though `_produce` never reached its CancelledError handler.
        for _ in range(20):
            if error_events:
                break
            await asyncio.sleep(0)
        assert len(error_events) == 1
        assert isinstance(error_events[0], asyncio.CancelledError)

    @pytest.mark.asyncio
    async def test_aclose_fires_on_llm_error_for_tracing(self) -> None:
        """Cancellation via `aclose()` must close the callback lifecycle.

        Without this, handlers / tracing see a started run with no
        matching end-or-error event for cancelled streams.
        """
        end_events: list[Any] = []
        error_events: list[BaseException] = []

        class RecordingHandler(AsyncCallbackHandler):
            async def on_llm_end(self, response: Any, **_: Any) -> None:
                end_events.append(response)

            async def on_llm_error(self, error: BaseException, **_: Any) -> None:
                error_events.append(error)

        handler = RecordingHandler()
        gate = asyncio.Event()
        model = _GatedStreamModel(gate=gate)
        stream = await model.astream_events(
            "test", config={"callbacks": [handler]}, version="v3"
        )

        aiter_ = stream.text.__aiter__()
        await aiter_.__anext__()

        await stream.aclose()

        # Let the shielded callback finish.
        for _ in range(10):
            if error_events:
                break
            await asyncio.sleep(0)

        assert not end_events, "on_llm_end must not fire for cancelled stream"
        assert len(error_events) == 1, (
            "aclose()-triggered cancellation must fire on_llm_error so "
            "tracing observes a matching end event."
        )
        assert isinstance(error_events[0], asyncio.CancelledError)

    @pytest.mark.asyncio
    async def test_aclose_preserves_successful_stream_mid_on_llm_end(self) -> None:
        """A successful stream must not be turned into CancelledError.

        After `message-finish` dispatches, `_output_proj` is already
        complete, but `_producer_task` may still be inside
        `run_manager.on_llm_end(...)`. Canceling unconditionally would
        drop the end callback and corrupt an otherwise successful run.
        """
        end_gate = asyncio.Event()
        end_fired = asyncio.Event()

        class SlowEndHandler(AsyncCallbackHandler):
            async def on_llm_end(self, response: Any, **_: Any) -> None:
                del response
                end_fired.set()
                await end_gate.wait()

        handler = SlowEndHandler()
        model = FakeListChatModel(responses=["ok"])
        stream = await model.astream_events(
            "test", config={"callbacks": [handler]}, version="v3"
        )

        # Wait until the stream has assembled the message and the
        # slow on_llm_end handler has started running.
        message = await stream.output
        await end_fired.wait()
        assert message.text == "ok"
        assert stream._producer_task is not None
        assert not stream._producer_task.done()
        assert stream._error is None

        # Kick off aclose; release the callback so it completes.
        close_task = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0)
        end_gate.set()
        await close_task

        assert stream._producer_task.done()
        assert not stream._producer_task.cancelled()
        # The success path must be preserved — no error installed.
        assert stream._error is None
        # And the output projection is still resolvable.
        assert (await stream.output).text == "ok"


class _V2RecordingHandler(BaseCallbackHandler, _V2StreamingCallbackHandler):
    """Records every protocol event dispatched via `on_stream_event`."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def on_stream_event(self, event: Any, **_: Any) -> None:
        self.events.append(event)


class _AsyncV2RecordingHandler(AsyncCallbackHandler, _V2StreamingCallbackHandler):
    """Async counterpart to `_V2RecordingHandler`."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def on_stream_event(self, event: Any, **_: Any) -> None:
        self.events.append(event)


class TestCacheHitV2Replay:
    """Cache hits must replay protocol events for v2 handlers.

    Without replay, `on_stream_event` fires on cache misses but not on
    warm-cache calls — LangGraph-style consumers would see behavior
    that depends on cache state alone.
    """

    def test_cache_hit_replays_events_to_v2_handler(self) -> None:
        cache = InMemoryCache()
        model = FakeListChatModel(responses=["Hello"], cache=cache)
        handler = _V2RecordingHandler()

        # Cold call: populates cache and fires events.
        model.invoke("prompt", config={"callbacks": [handler]})
        cold_events = list(handler.events)
        handler.events.clear()

        # Warm call: events must fire again from the replayed cache hit.
        model.invoke("prompt", config={"callbacks": [handler]})
        warm_events = list(handler.events)

        assert warm_events, "cache hit must replay v2 events"
        warm_types = [e["event"] for e in warm_events]
        # Lifecycle anchors must be present on the warm path, matching cold.
        # Replay collapses per-chunk deltas into a single delta per block,
        # so we assert shape equivalence at the anchor level rather than
        # exact event-count equality.
        assert warm_types[0] == "message-start"
        assert warm_types[-1] == "message-finish"
        assert "content-block-start" in warm_types
        assert "content-block-finish" in warm_types
        cold_types = [e["event"] for e in cold_events]
        assert cold_types[0] == "message-start"
        assert cold_types[-1] == "message-finish"

    def test_cache_hit_skips_replay_without_v2_handler(self) -> None:
        """A v1-only callback set must not accidentally trigger v2 replay."""
        cache = InMemoryCache()
        model = FakeListChatModel(responses=["Hi"], cache=cache)

        # Prime the cache.
        model.invoke("prompt")

        class _V1OnlyHandler(BaseCallbackHandler):
            def __init__(self) -> None:
                self.stream_events: list[Any] = []

            def on_stream_event(self, event: Any, **_: Any) -> None:
                self.stream_events.append(event)

        handler = _V1OnlyHandler()
        model.invoke("prompt", config={"callbacks": [handler]})
        # No `_V2StreamingCallbackHandler` marker -> no replay.
        assert handler.stream_events == []

    @pytest.mark.asyncio
    async def test_acache_hit_replays_events_to_v2_handler(self) -> None:
        cache = InMemoryCache()
        model = FakeListChatModel(responses=["Hello"], cache=cache)
        handler = _AsyncV2RecordingHandler()

        await model.ainvoke("prompt", config={"callbacks": [handler]})
        handler.events.clear()

        await model.ainvoke("prompt", config={"callbacks": [handler]})
        assert handler.events, "async cache hit must replay v2 events"
        types = [e["event"] for e in handler.events]
        assert "message-start" in types
        assert "message-finish" in types


class _ProviderMetadataStreamModel(BaseChatModel):
    """Fake model that advertises `output_version="responses/v1"` in metadata.

    Verifies that `stream_events(version="v3")` pins the assembled
    message's `output_version` to `"v1"` — the shape it actually
    produces — regardless of what the provider's chunk metadata claims.
    """

    @property
    def _llm_type(self) -> str:
        return "provider-metadata-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise NotImplementedError

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="hi",
                response_metadata={"output_version": "responses/v1"},
            )
        )


class TestOutputVersionPinning:
    """`stream_events(version="v3").output` always serializes as v1 content blocks."""

    def test_output_version_pinned_to_v1(self) -> None:
        model = _ProviderMetadataStreamModel()
        stream = model.stream_events("hi", version="v3")
        msg = stream.output
        # Assembled message must claim `"v1"` even though the provider
        # chunk metadata advertised `"responses/v1"`.
        assert msg.response_metadata.get("output_version") == "v1"


class _BedrockConverseToolCallModel(BaseChatModel):
    """Replays a captured `ChatBedrockConverse` tool-calling stream.

    Bedrock opens a tool block with `args=None` (name + id only) and
    starts streaming JSON args in the *next* chunk. Other providers
    emit `args=""` on the opener, so the compat bridge's accumulator
    never saw `None` on the state side until Bedrock hit it.
    """

    @property
    def _llm_type(self) -> str:
        return "bedrock-converse-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise NotImplementedError

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        meta = {"model_provider": "bedrock_converse", "ls_provider": "amazon_bedrock"}

        # Text at content index 0.
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=[{"type": "text", "text": "Hello ", "index": 0}],
                response_metadata=meta,
            )
        )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=[{"type": "text", "text": "Boston!", "index": 0}],
                response_metadata=meta,
            )
        )
        # Tool opener: name + id, args=None (the Bedrock-specific shape).
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {
                        "type": "tool_use",
                        "name": "get_weather",
                        "id": "tooluse_1",
                        "index": 1,
                    }
                ],
                response_metadata=meta,
                tool_call_chunks=[
                    {
                        "name": "get_weather",
                        "args": None,
                        "id": "tooluse_1",
                        "index": 1,
                        "type": "tool_call_chunk",
                    }
                ],
            )
        )
        # Args deltas; each intermediate JSON slice is itself unparseable.
        for slice_ in ('{"location":', ' "Boston', '"}'):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=[
                        {"type": "tool_use", "input": slice_, "id": None, "index": 1}
                    ],
                    response_metadata=meta,
                    tool_call_chunks=[
                        {
                            "name": None,
                            "args": slice_,
                            "id": None,
                            "index": 1,
                            "type": "tool_call_chunk",
                        }
                    ],
                )
            )
        # Terminal chunk carrying usage + stop reason.
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={"stopReason": "tool_use", **meta},
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                chunk_position="last",
            )
        )


class TestBedrockConverseToolCallArgs:
    """Regression: Bedrock's `args=None` tool opener must not break accumulation.

    The compat bridge's `_accumulate` used to do
    `state.get("args", "") + (delta.get("args") or "")`; `state.get("args", "")`
    returns the stored value when the key exists, so a Bedrock opener that
    stores `args=None` poisoned the state and the next delta raised
    `TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'`.
    """

    def test_bedrock_tool_call_assembles_without_error(self) -> None:
        model = _BedrockConverseToolCallModel()
        stream = model.stream_events("What's the weather in Boston?", version="v3")
        # Drive the stream to completion — the raise would have surfaced here.
        events = list(stream)

        kinds = [e["event"] for e in events]
        assert kinds[0] == "message-start"
        assert kinds[-1] == "message-finish"

        msg = stream.output
        assert msg.tool_calls == [
            {
                "name": "get_weather",
                "args": {"location": "Boston"},
                "id": "tooluse_1",
                "type": "tool_call",
            }
        ]
        # The args are assembled by concatenating deltas, so no
        # partial-JSON slice should register as an `invalid_tool_call`.
        assert msg.invalid_tool_calls == []
        # Text block round-trips alongside the tool call.
        text_blocks = [
            b for b in msg.content if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Hello Boston!"
