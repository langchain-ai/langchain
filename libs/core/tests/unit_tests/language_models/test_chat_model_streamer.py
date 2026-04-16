"""Tests for BaseChatModel.stream_v2() / astream_v2()."""

from __future__ import annotations

from typing import Any

import pytest

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    ChatModelStream,
)
from langchain_core.language_models.fake_chat_models import FakeListChatModel


class TestStreamV2Sync:
    """Test BaseChatModel.stream_v2() with FakeListChatModel."""

    def test_stream_text(self) -> None:
        model = FakeListChatModel(responses=["Hello world!"])
        stream = model.stream_v2("test")

        assert isinstance(stream, ChatModelStream)
        deltas = list(stream.text)
        assert "".join(deltas) == "Hello world!"
        assert stream.done

    def test_stream_output(self) -> None:
        model = FakeListChatModel(responses=["Hello!"])
        stream = model.stream_v2("test")

        msg = stream.output
        assert msg.content == "Hello!"
        assert msg.id is not None

    def test_stream_usage_none_for_fake(self) -> None:
        model = FakeListChatModel(responses=["Hi"])
        stream = model.stream_v2("test")
        # Drain
        for _ in stream.text:
            pass
        assert stream.usage is None

    def test_stream_raw_events(self) -> None:
        model = FakeListChatModel(responses=["ab"])
        stream = model.stream_v2("test")

        events = list(stream)
        event_types = [e.get("event") for e in events]
        assert event_types[0] == "message-start"
        assert event_types[-1] == "message-finish"
        assert "content-block-delta" in event_types


class TestAstreamV2:
    """Test BaseChatModel.astream_v2() with FakeListChatModel."""

    @pytest.mark.asyncio
    async def test_astream_text_await(self) -> None:
        model = FakeListChatModel(responses=["Hello!"])
        stream = await model.astream_v2("test")

        assert isinstance(stream, AsyncChatModelStream)
        full = await stream.text
        assert full == "Hello!"

    @pytest.mark.asyncio
    async def test_astream_text_deltas(self) -> None:
        model = FakeListChatModel(responses=["Hi"])
        stream = await model.astream_v2("test")

        deltas = [d async for d in stream.text]
        assert "".join(deltas) == "Hi"

    @pytest.mark.asyncio
    async def test_astream_await_output(self) -> None:
        model = FakeListChatModel(responses=["Hey"])
        stream = await model.astream_v2("test")

        msg = await stream
        assert msg.content == "Hey"


class _RecordingHandler(BaseCallbackHandler):
    """Sync callback handler that records lifecycle hook invocations."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("on_chat_model_start")

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("on_llm_end")

    def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("on_llm_error")


class _AsyncRecordingHandler(AsyncCallbackHandler):
    """Async callback handler that records lifecycle hook invocations."""

    def __init__(self) -> None:
        self.events: list[str] = []

    async def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("on_chat_model_start")

    async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("on_llm_end")

    async def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("on_llm_error")


class TestCallbacks:
    """Verify stream_v2 fires on_llm_end / on_llm_error callbacks."""

    def test_on_llm_end_fires_after_drain(self) -> None:
        handler = _RecordingHandler()
        model = FakeListChatModel(responses=["done"], callbacks=[handler])
        stream = model.stream_v2("test")
        for _ in stream.text:
            pass
        _ = stream.output

        assert "on_chat_model_start" in handler.events
        assert "on_llm_end" in handler.events
        assert handler.events.index("on_llm_end") > handler.events.index(
            "on_chat_model_start"
        )

    @pytest.mark.asyncio
    async def test_on_llm_end_fires_async(self) -> None:
        handler = _AsyncRecordingHandler()
        model = FakeListChatModel(responses=["done"], callbacks=[handler])
        stream = await model.astream_v2("test")
        _ = await stream

        assert "on_chat_model_start" in handler.events
        assert "on_llm_end" in handler.events
