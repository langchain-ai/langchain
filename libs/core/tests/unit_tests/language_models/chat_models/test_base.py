"""Test base chat model."""

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import pytest

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModelError
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.tracers import LogStreamCallbackHandler
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.event_stream import _AstreamEventsCallbackHandler
from langchain_core.tracers.schemas import Run
from tests.unit_tests.fake.callbacks import (
    BaseFakeCallbackHandler,
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)
from tests.unit_tests.stubs import _any_id_ai_message, _any_id_ai_message_chunk

if TYPE_CHECKING:
    from langchain_core.outputs.llm_result import LLMResult


@pytest.fixture
def messages() -> list:
    return [
        SystemMessage(content="You are a test user."),
        HumanMessage(content="Hello, I am a test user."),
    ]


@pytest.fixture
def messages_2() -> list:
    return [
        SystemMessage(content="You are a test user."),
        HumanMessage(content="Hello, I not a test user."),
    ]


def test_batch_size(messages: list, messages_2: list) -> None:
    # The base endpoint doesn't support native batching,
    # so we expect batch_size to always be 1
    llm = FakeListChatModel(responses=[str(i) for i in range(100)])
    with collect_runs() as cb:
        llm.batch([messages, messages_2], {"callbacks": [cb]})
        assert len(cb.traced_runs) == 2
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
    with collect_runs() as cb:
        llm.batch([messages], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 1

    with collect_runs() as cb:
        llm.invoke(messages)
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    with collect_runs() as cb:
        list(llm.stream(messages))
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1


async def test_async_batch_size(messages: list, messages_2: list) -> None:
    llm = FakeListChatModel(responses=[str(i) for i in range(100)])
    # The base endpoint doesn't support native batching,
    # so we expect batch_size to always be 1
    with collect_runs() as cb:
        await llm.abatch([messages, messages_2], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 2
    with collect_runs() as cb:
        await llm.abatch([messages], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 1

    with collect_runs() as cb:
        await llm.ainvoke(messages)
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    with collect_runs() as cb:
        async for _ in llm.astream(messages):
            pass
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1


async def test_stream_error_callback() -> None:
    message = "test"

    def eval_response(callback: BaseFakeCallbackHandler, i: int) -> None:
        assert callback.errors == 1
        assert len(callback.errors_args) == 1
        llm_result: LLMResult = callback.errors_args[0]["kwargs"]["response"]
        if i == 0:
            assert llm_result.generations == []
        else:
            assert llm_result.generations[0][0].text == message[:i]

    for i in range(2):
        llm = FakeListChatModel(
            responses=[message],
            error_on_chunk_number=i,
        )
        with pytest.raises(FakeListChatModelError):
            cb_async = FakeAsyncCallbackHandler()
            async for _ in llm.astream("Dummy message", callbacks=[cb_async]):
                pass
            eval_response(cb_async, i)

            cb_sync = FakeCallbackHandler()
            for _ in llm.stream("Dumy message", callbacks=[cb_sync]):
                pass

            eval_response(cb_sync, i)


async def test_astream_fallback_to_ainvoke() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithGenerate(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Top Level call."""
            message = AIMessage(content="hello")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithGenerate()
    chunks = list(model.stream("anything"))
    assert chunks == [_any_id_ai_message(content="hello")]

    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == [_any_id_ai_message(content="hello")]


async def test_astream_implementation_fallback_to_stream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithSyncStream(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Top Level call."""
            raise NotImplementedError

        def _stream(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> Iterator[ChatGenerationChunk]:
            """Stream the output of the model."""
            yield ChatGenerationChunk(message=AIMessageChunk(content="a"))
            yield ChatGenerationChunk(message=AIMessageChunk(content="b"))

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithSyncStream()
    chunks = list(model.stream("anything"))
    assert chunks == [
        _any_id_ai_message_chunk(content="a"),
        _any_id_ai_message_chunk(content="b"),
    ]
    assert len({chunk.id for chunk in chunks}) == 1
    assert type(model)._astream == BaseChatModel._astream
    astream_chunks = [chunk async for chunk in model.astream("anything")]
    assert astream_chunks == [
        _any_id_ai_message_chunk(content="a"),
        _any_id_ai_message_chunk(content="b"),
    ]
    assert len({chunk.id for chunk in astream_chunks}) == 1


async def test_astream_implementation_uses_astream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithAsyncStream(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Top Level call."""
            raise NotImplementedError

        async def _astream(  # type: ignore
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> AsyncIterator[ChatGenerationChunk]:
            """Stream the output of the model."""
            yield ChatGenerationChunk(message=AIMessageChunk(content="a"))
            yield ChatGenerationChunk(message=AIMessageChunk(content="b"))

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithAsyncStream()
    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == [
        _any_id_ai_message_chunk(content="a"),
        _any_id_ai_message_chunk(content="b"),
    ]
    assert len({chunk.id for chunk in chunks}) == 1


class FakeTracer(BaseTracer):
    def __init__(self) -> None:
        super().__init__()
        self.traced_run_ids: list = []

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""
        self.traced_run_ids.append(run.id)


def test_pass_run_id() -> None:
    llm = FakeListChatModel(responses=["a", "b", "c"])
    cb = FakeTracer()
    uid1 = uuid.uuid4()
    llm.invoke("Dummy message", {"callbacks": [cb], "run_id": uid1})
    assert cb.traced_run_ids == [uid1]
    uid2 = uuid.uuid4()
    list(llm.stream("Dummy message", {"callbacks": [cb], "run_id": uid2}))
    assert cb.traced_run_ids == [uid1, uid2]
    uid3 = uuid.uuid4()
    llm.batch([["Dummy message"]], {"callbacks": [cb], "run_id": uid3})
    assert cb.traced_run_ids == [uid1, uid2, uid3]


async def test_async_pass_run_id() -> None:
    llm = FakeListChatModel(responses=["a", "b", "c"])
    cb = FakeTracer()
    uid1 = uuid.uuid4()
    await llm.ainvoke("Dummy message", {"callbacks": [cb], "run_id": uid1})
    assert cb.traced_run_ids == [uid1]
    uid2 = uuid.uuid4()
    async for _ in llm.astream("Dummy message", {"callbacks": [cb], "run_id": uid2}):
        pass
    assert cb.traced_run_ids == [uid1, uid2]

    uid3 = uuid.uuid4()
    await llm.abatch([["Dummy message"]], {"callbacks": [cb], "run_id": uid3})
    assert cb.traced_run_ids == [uid1, uid2, uid3]


class NoStreamingModel(BaseChatModel):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("invoke"))])

    @property
    def _llm_type(self) -> str:
        return "model1"


class StreamingModel(NoStreamingModel):
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="stream"))


@pytest.mark.parametrize("disable_streaming", [True, False, "tool_calling"])
def test_disable_streaming(
    disable_streaming: Union[bool, Literal["tool_calling"]],
) -> None:
    model = StreamingModel(disable_streaming=disable_streaming)
    assert model.invoke([]).content == "invoke"

    expected = "invoke" if disable_streaming is True else "stream"
    assert next(model.stream([])).content == expected
    assert (
        model.invoke([], config={"callbacks": [LogStreamCallbackHandler()]}).content
        == expected
    )

    expected = "invoke" if disable_streaming in ("tool_calling", True) else "stream"
    assert next(model.stream([], tools=[{"type": "function"}])).content == expected
    assert (
        model.invoke(
            [], config={"callbacks": [LogStreamCallbackHandler()]}, tools=[{}]
        ).content
        == expected
    )


@pytest.mark.parametrize("disable_streaming", [True, False, "tool_calling"])
async def test_disable_streaming_async(
    disable_streaming: Union[bool, Literal["tool_calling"]],
) -> None:
    model = StreamingModel(disable_streaming=disable_streaming)
    assert (await model.ainvoke([])).content == "invoke"

    expected = "invoke" if disable_streaming is True else "stream"
    async for c in model.astream([]):
        assert c.content == expected
        break
    assert (
        await model.ainvoke([], config={"callbacks": [_AstreamEventsCallbackHandler()]})
    ).content == expected

    expected = "invoke" if disable_streaming in ("tool_calling", True) else "stream"
    async for c in model.astream([], tools=[{}]):
        assert c.content == expected
        break
    assert (
        await model.ainvoke(
            [], config={"callbacks": [_AstreamEventsCallbackHandler()]}, tools=[{}]
        )
    ).content == expected


@pytest.mark.parametrize("disable_streaming", [True, False, "tool_calling"])
def test_disable_streaming_no_streaming_model(
    disable_streaming: Union[bool, Literal["tool_calling"]],
) -> None:
    model = NoStreamingModel(disable_streaming=disable_streaming)
    assert model.invoke([]).content == "invoke"
    assert next(model.stream([])).content == "invoke"
    assert (
        model.invoke([], config={"callbacks": [LogStreamCallbackHandler()]}).content
        == "invoke"
    )
    assert next(model.stream([], tools=[{}])).content == "invoke"


@pytest.mark.parametrize("disable_streaming", [True, False, "tool_calling"])
async def test_disable_streaming_no_streaming_model_async(
    disable_streaming: Union[bool, Literal["tool_calling"]],
) -> None:
    model = NoStreamingModel(disable_streaming=disable_streaming)
    assert (await model.ainvoke([])).content == "invoke"
    async for c in model.astream([]):
        assert c.content == "invoke"
        break
    assert (
        await model.ainvoke([], config={"callbacks": [_AstreamEventsCallbackHandler()]})
    ).content == "invoke"
    async for c in model.astream([], tools=[{}]):
        assert c.content == "invoke"
        break
