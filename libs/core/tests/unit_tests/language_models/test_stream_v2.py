"""Tests for stream_v2 / astream_v2 and ChatModelStream."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from langchain_protocol.protocol import (
    ContentBlockDeltaData,
    ContentBlockFinishData,
    MessageFinishData,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    UsageInfo,
)

from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    ChatModelStream,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult

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


class TestChatModelStream:
    """Test the sync ChatModelStream object."""

    def test_push_text_delta(self) -> None:
        stream = ChatModelStream()
        stream._push_content_block_delta(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                content_block=TextBlock(type="text", text="Hello"),
            )
        )
        assert stream._text_acc == "Hello"

    def test_push_reasoning_delta(self) -> None:
        stream = ChatModelStream()
        stream._push_content_block_delta(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                content_block=ReasoningBlock(type="reasoning", reasoning="think"),
            )
        )
        assert stream._reasoning_acc == "think"

    def test_push_content_block_finish_tool_call(self) -> None:
        stream = ChatModelStream()
        stream._push_content_block_finish(
            ContentBlockFinishData(
                event="content-block-finish",
                index=0,
                content_block=ToolCallBlock(
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
        stream._finish(
            MessageFinishData(event="message-finish", reason="stop", usage=usage)
        )
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
                content_block=TextBlock(type="text", text="Hi"),
            ),
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                content_block=TextBlock(type="text", text=" there"),
            ),
        ]
        finish = MessageFinishData(event="message-finish", reason="stop")
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
                content_block=TextBlock(type="text", text="Hello"),
            )
        )
        stream._push_content_block_delta(
            ContentBlockDeltaData(
                event="content-block-delta",
                index=0,
                content_block=TextBlock(type="text", text=" world"),
            )
        )
        stream._finish(MessageFinishData(event="message-finish", reason="stop"))

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
                    content_block=TextBlock(type="text", text="a"),
                )
            )
            await asyncio.sleep(0)
            stream._push_content_block_delta(
                ContentBlockDeltaData(
                    event="content-block-delta",
                    index=0,
                    content_block=TextBlock(type="text", text="b"),
                )
            )
            await asyncio.sleep(0)
            stream._finish(MessageFinishData(event="message-finish", reason="stop"))

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
                content_block=ToolCallBlock(
                    type="tool_call",
                    id="tc1",
                    name="search",
                    args={"q": "test"},
                ),
            )
        )
        stream._finish(MessageFinishData(event="message-finish", reason="tool_use"))

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
    """Test BaseChatModel.stream_v2() with FakeListChatModel."""

    def test_stream_v2_text(self) -> None:
        model = FakeListChatModel(responses=["Hello world!"])
        stream = model.stream_v2("test")

        assert isinstance(stream, ChatModelStream)
        deltas = list(stream.text)
        assert "".join(deltas) == "Hello world!"
        assert stream.done

    def test_stream_v2_usage(self) -> None:
        model = FakeListChatModel(responses=["Hi"])
        stream = model.stream_v2("test")

        # Drain stream
        for _ in stream.text:
            pass
        # FakeListChatModel doesn't emit usage, so it should be None
        assert stream.usage is None
        assert stream.done

    def test_stream_v2_malformed_tool_args_produce_invalid_tool_call(self) -> None:
        """End-to-end: malformed tool-call JSON becomes invalid_tool_calls."""
        model = _MalformedToolCallModel()
        stream = model.stream_v2("test")
        msg = stream.output

        assert msg.tool_calls == []
        assert len(msg.invalid_tool_calls) == 1
        itc = msg.invalid_tool_calls[0]
        assert itc["name"] == "search"
        assert itc["args"] == '{"q": '
        assert itc["id"] == "call_1"


class TestAstreamV2:
    """Test BaseChatModel.astream_v2() with FakeListChatModel."""

    @pytest.mark.asyncio
    async def test_astream_v2_text(self) -> None:
        model = FakeListChatModel(responses=["Hello!"])
        stream = await model.astream_v2("test")

        assert isinstance(stream, AsyncChatModelStream)
        full = await stream.text
        assert full == "Hello!"

    @pytest.mark.asyncio
    async def test_astream_v2_deltas(self) -> None:
        model = FakeListChatModel(responses=["Hi"])
        stream = await model.astream_v2("test")

        deltas = [d async for d in stream.text]
        assert "".join(deltas) == "Hi"
