"""Tests for `BaseChatModel.stream_events(version="v3")` and its async equivalent."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import Field

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    ChatModelStream,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Iterator

    from langchain_protocol.protocol import MessagesData

    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult


class TestStreamV2Sync:
    """Test `BaseChatModel.stream_events(version="v3")` with `FakeListChatModel`."""

    def test_stream_text(self) -> None:
        model = FakeListChatModel(responses=["Hello world!"])
        stream = model.stream_events("test", version="v3")

        assert isinstance(stream, ChatModelStream)
        deltas = list(stream.text)
        assert "".join(deltas) == "Hello world!"
        assert stream.done

    def test_stream_output(self) -> None:
        model = FakeListChatModel(responses=["Hello!"])
        stream = model.stream_events("test", version="v3")

        msg = stream.output
        assert isinstance(msg.content, list)
        assert msg.content == [{"type": "text", "text": "Hello!", "index": 0}]
        assert msg.id is not None

    def test_stream_usage_none_for_fake(self) -> None:
        model = FakeListChatModel(responses=["Hi"])
        stream = model.stream_events("test", version="v3")
        # Drain
        for _ in stream.text:
            pass
        assert stream.output.usage_metadata is None

    def test_stream_raw_events(self) -> None:
        model = FakeListChatModel(responses=["ab"])
        stream = model.stream_events("test", version="v3")

        events = list(stream)
        event_types = [e.get("event") for e in events]
        assert event_types[0] == "message-start"
        assert event_types[-1] == "message-finish"
        assert "content-block-delta" in event_types


class TestAstreamV2:
    """Test `BaseChatModel.astream_events(version="v3")` with `FakeListChatModel`."""

    @pytest.mark.asyncio
    async def test_astream_text_await(self) -> None:
        model = FakeListChatModel(responses=["Hello!"])
        stream = await model.astream_events("test", version="v3")

        assert isinstance(stream, AsyncChatModelStream)
        full = await stream.text
        assert full == "Hello!"

    @pytest.mark.asyncio
    async def test_astream_text_deltas(self) -> None:
        model = FakeListChatModel(responses=["Hi"])
        stream = await model.astream_events("test", version="v3")

        deltas = [d async for d in stream.text]
        assert "".join(deltas) == "Hi"

    @pytest.mark.asyncio
    async def test_astream_await_output(self) -> None:
        model = FakeListChatModel(responses=["Hey"])
        stream = await model.astream_events("test", version="v3")

        msg = await stream
        assert msg.content == [{"type": "text", "text": "Hey", "index": 0}]


class _RecordingHandler(BaseCallbackHandler):
    """Sync callback handler that records lifecycle hook invocations."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.stream_events: list[MessagesData] = []
        self.last_llm_end_response: LLMResult | None = None

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("on_chat_model_start")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        del kwargs
        self.events.append("on_llm_end")
        self.last_llm_end_response = response

    def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("on_llm_error")

    def on_stream_event(self, event: MessagesData, **kwargs: Any) -> None:
        del kwargs
        self.stream_events.append(event)


class _AsyncRecordingHandler(AsyncCallbackHandler):
    """Async callback handler that records lifecycle hook invocations."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.stream_events: list[MessagesData] = []
        self.last_llm_end_response: LLMResult | None = None

    async def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("on_chat_model_start")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        del kwargs
        self.events.append("on_llm_end")
        self.last_llm_end_response = response

    async def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("on_llm_error")

    async def on_stream_event(self, event: MessagesData, **kwargs: Any) -> None:
        del kwargs
        self.stream_events.append(event)


class _EmptyStreamModel(BaseChatModel):
    """Fake chat model whose stream producers yield no chunks."""

    @property
    def _llm_type(self) -> str:
        return "empty-stream-fake"

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
        if False:
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        if False:
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))


class TestCallbacks:
    """Verify v3 streaming fires `on_llm_end` / `on_llm_error` callbacks."""

    def test_stream_events_v3_defers_on_chat_model_start_until_consumed(self) -> None:
        handler = _RecordingHandler()
        model = FakeListChatModel(responses=["done"], callbacks=[handler])

        stream = model.stream_events("test", version="v3")

        assert handler.events == []

        _ = stream.output

        assert handler.events[0] == "on_chat_model_start"

    def test_on_llm_end_fires_after_drain(self) -> None:
        handler = _RecordingHandler()
        model = FakeListChatModel(responses=["done"], callbacks=[handler])
        stream = model.stream_events("test", version="v3")
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
        stream = await model.astream_events("test", version="v3")
        _ = await stream

        assert "on_chat_model_start" in handler.events
        assert "on_llm_end" in handler.events

    @pytest.mark.asyncio
    async def test_astream_events_v3_defers_on_chat_model_start_until_consumed(
        self,
    ) -> None:
        handler = _AsyncRecordingHandler()
        model = FakeListChatModel(responses=["done"], callbacks=[handler])

        stream = await model.astream_events("test", version="v3")

        assert handler.events == []

        _ = await stream

        assert handler.events[0] == "on_chat_model_start"

    def test_on_llm_end_receives_assembled_message(self) -> None:
        """The LLMResult passed to on_llm_end must carry the final message.

        Without this, LangSmith traces would see an empty generations list.
        """
        handler = _RecordingHandler()
        model = FakeListChatModel(responses=["hello"], callbacks=[handler])
        stream = model.stream_events("test", version="v3")
        _ = stream.output

        response = handler.last_llm_end_response
        assert response is not None
        assert response.generations
        gen = response.generations[0][0]
        assert isinstance(gen, ChatGeneration)
        assert gen.message.content == [{"type": "text", "text": "hello", "index": 0}]

    @pytest.mark.asyncio
    async def test_on_llm_end_receives_assembled_message_async(self) -> None:
        handler = _AsyncRecordingHandler()
        model = FakeListChatModel(responses=["hello"], callbacks=[handler])
        stream = await model.astream_events("test", version="v3")
        _ = await stream

        response = handler.last_llm_end_response
        assert response is not None
        assert response.generations
        gen = response.generations[0][0]
        assert isinstance(gen, ChatGeneration)
        assert gen.message.content == [{"type": "text", "text": "hello", "index": 0}]

    def test_empty_stream_reports_error_without_finish_only_lifecycle(self) -> None:
        handler = _RecordingHandler()
        stream = _EmptyStreamModel(callbacks=[handler]).stream_events(
            "test", version="v3"
        )

        with pytest.raises(ValueError, match="No generation chunks were returned"):
            list(stream)

        assert handler.stream_events == []
        assert "on_llm_error" in handler.events
        assert "on_llm_end" not in handler.events

    @pytest.mark.asyncio
    async def test_empty_astream_reports_error(self) -> None:
        handler = _AsyncRecordingHandler()
        stream = await _EmptyStreamModel(callbacks=[handler]).astream_events(
            "test", version="v3"
        )

        with pytest.raises(ValueError, match="No generation chunks were returned"):
            await stream
        task = stream._producer_task
        assert task is not None
        await task

        assert handler.stream_events == []
        assert "on_llm_error" in handler.events
        assert "on_llm_end" not in handler.events


class TestOnStreamEvent:
    """`on_stream_event` fires once per protocol event from v3 streaming."""

    def test_on_stream_event_fires_for_every_event_sync(self) -> None:
        handler = _RecordingHandler()
        model = FakeListChatModel(responses=["Hi"], callbacks=[handler])
        stream = model.stream_events("test", version="v3")
        _ = stream.output

        # Every event the stream sees should also reach the observer.
        assert len(handler.stream_events) == len(list(stream))
        event_types = [e["event"] for e in handler.stream_events]
        assert event_types[0] == "message-start"
        assert event_types[-1] == "message-finish"
        assert "content-block-delta" in event_types

    @pytest.mark.asyncio
    async def test_on_stream_event_fires_for_every_event_async(self) -> None:
        handler = _AsyncRecordingHandler()
        model = FakeListChatModel(responses=["Hi"], callbacks=[handler])
        stream = await model.astream_events("test", version="v3")
        _ = await stream

        event_types = [e["event"] for e in handler.stream_events]
        assert event_types[0] == "message-start"
        assert event_types[-1] == "message-finish"
        assert "content-block-delta" in event_types

    def test_on_stream_event_ordering_relative_to_lifecycle(self) -> None:
        """Stream events must all fire between on_chat_model_start and on_llm_end."""
        handler = _RecordingHandler()
        model = FakeListChatModel(responses=["Hi"], callbacks=[handler])
        stream = model.stream_events("test", version="v3")
        _ = stream.output

        # on_stream_event doesn't show up in `events` (different list), but
        # on_chat_model_start and on_llm_end bracket the run.
        assert handler.events[0] == "on_chat_model_start"
        assert handler.events[-1] == "on_llm_end"
        # And we did see stream events during that bracket.
        assert handler.stream_events


class TestCancellation:
    """Cancellation of `astream_events(version="v3")` must propagate."""

    @pytest.mark.asyncio
    async def test_astream_events_v3_cancellation_propagates(self) -> None:
        """Cancelling the producer task must raise CancelledError.

        Regression test: the producer's `except BaseException` previously
        swallowed `asyncio.CancelledError`, converting it into an
        `on_llm_error` + `stream._fail` pair that never propagated.
        """
        model = FakeListChatModel(responses=["abcdefghij"], sleep=0.05)
        stream = await model.astream_events("test", version="v3")
        aiter_ = stream.text.__aiter__()
        await aiter_.__anext__()
        task = stream._producer_task
        assert task is not None

        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert isinstance(stream._error, asyncio.CancelledError)


class _KwargRecordingModel(FakeListChatModel):
    """Fake model that records kwargs passed to `_stream` / `_astream`."""

    received_kwargs: list[dict[str, Any]] = Field(default_factory=list)

    def _stream(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.received_kwargs.append({"stop": stop, **kwargs})
        return super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.received_kwargs.append({"stop": stop, **kwargs})
        async for chunk in super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk


class TestRunnableBindingForwarding:
    """`RunnableBinding.stream_events(version="v3")` merges bound kwargs.

    Without the explicit override on `RunnableBinding`, `__getattr__`
    forwards the call but drops `self.kwargs` — so tools bound via
    `bind_tools`, stop sequences bound via `bind`, etc. would be silently
    ignored.
    """

    def test_bound_kwargs_reach_stream_events_v3(self) -> None:
        model = _KwargRecordingModel(responses=["hi"])
        model.received_kwargs = []
        bound = model.bind(my_marker="sentinel-42")

        stream = bound.stream_events("test", version="v3")
        for _ in stream.text:
            pass

        assert len(model.received_kwargs) == 1
        assert model.received_kwargs[0].get("my_marker") == "sentinel-42"

    def test_call_kwargs_override_bound_kwargs(self) -> None:
        model = _KwargRecordingModel(responses=["hi"])
        model.received_kwargs = []
        bound = model.bind(my_marker="from-bind")

        stream = bound.stream_events("test", my_marker="from-call", version="v3")
        for _ in stream.text:
            pass

        assert model.received_kwargs[0].get("my_marker") == "from-call"

    @pytest.mark.asyncio
    async def test_bound_kwargs_reach_astream_events_v3(self) -> None:
        model = _KwargRecordingModel(responses=["hi"])
        model.received_kwargs = []
        bound = model.bind(my_marker="sentinel-async")

        stream = await bound.astream_events("test", version="v3")
        _ = await stream

        assert len(model.received_kwargs) == 1
        assert model.received_kwargs[0].get("my_marker") == "sentinel-async"

    def test_bound_version_routes_to_v3_without_call_site_repeat(self) -> None:
        # `bind(version="v3").stream_events(input)` must route to the v3
        # branch (using the bound `version`) and must not forward `version`
        # to the underlying model as an extra kwarg.
        model = _KwargRecordingModel(responses=["hi"])
        model.received_kwargs = []
        bound = model.bind(version="v3")

        # `version` is in `self.kwargs`, not at the call site, so the
        # static return type is the v1/v2 iterator overload — narrow it.
        stream = cast("ChatModelStream", bound.stream_events("test"))
        chunks = list(stream.text)

        assert "".join(chunks) == "hi"
        assert len(model.received_kwargs) == 1
        assert "version" not in model.received_kwargs[0]

    @pytest.mark.asyncio
    async def test_bound_version_routes_to_v3_async_without_call_site_repeat(
        self,
    ) -> None:
        model = _KwargRecordingModel(responses=["hi"])
        model.received_kwargs = []
        bound = model.bind(version="v3")

        stream = await cast(
            "Awaitable[AsyncChatModelStream]", bound.astream_events("test")
        )
        _ = await stream

        assert len(model.received_kwargs) == 1
        assert "version" not in model.received_kwargs[0]
