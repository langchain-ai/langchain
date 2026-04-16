"""Tests for stream_v2 / astream_v2 and ChatModelStream."""

import asyncio

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
from langchain_core.language_models.fake_chat_models import FakeListChatModel


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
        stream._fail(RuntimeError("test"))
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

        stream._bind_pump(pump_one)

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
        stream._fail(RuntimeError("boom"))

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
