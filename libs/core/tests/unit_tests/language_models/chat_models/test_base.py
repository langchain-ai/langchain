"""Test base chat model."""

from typing import Any, AsyncIterator, Iterator, List, Optional

import pytest

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.tracers.context import collect_runs
from tests.unit_tests.fake.callbacks import (
    BaseFakeCallbackHandler,
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)


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
        assert all([(r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs])
    with collect_runs() as cb:
        llm.batch([messages], {"callbacks": [cb]})
        assert all([(r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs])
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
        assert all([(r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs])
        assert len(cb.traced_runs) == 2
    with collect_runs() as cb:
        await llm.abatch([messages], {"callbacks": [cb]})
        assert all([(r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs])
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

    for i in range(0, 2):
        llm = FakeListChatModel(
            responses=[message],
            error_on_chunk_number=i,
        )
        with pytest.raises(Exception):
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
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Top Level call"""
            message = AIMessage(content="hello")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithGenerate()
    chunks = [chunk for chunk in model.stream("anything")]
    assert chunks == [AIMessage(content="hello")]

    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == [AIMessage(content="hello")]


async def test_astream_implementation_fallback_to_stream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithSyncStream(BaseChatModel):
        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Top Level call"""
            raise NotImplementedError()

        def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
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
    chunks = [chunk for chunk in model.stream("anything")]
    assert chunks == [
        AIMessageChunk(content="a"),
        AIMessageChunk(content="b"),
    ]
    assert type(model)._astream == BaseChatModel._astream
    astream_chunks = [chunk async for chunk in model.astream("anything")]
    assert astream_chunks == [
        AIMessageChunk(content="a"),
        AIMessageChunk(content="b"),
    ]


async def test_astream_implementation_uses_astream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithAsyncStream(BaseChatModel):
        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Top Level call"""
            raise NotImplementedError()

        async def _astream(  # type: ignore
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
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
        AIMessageChunk(content="a"),
        AIMessageChunk(content="b"),
    ]
