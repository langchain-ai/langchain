"""Test base chat model."""

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import pytest
from typing_extensions import override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import (
    BaseChatModel,
    FakeListChatModel,
    ParrotFakeChatModel,
)
from langchain_core.language_models._utils import _normalize_messages
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


@pytest.mark.xfail(reason="This test is failing due to a bug in the testing code")
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

    for i in range(len(message)):
        llm = FakeListChatModel(
            responses=[message],
            error_on_chunk_number=i,
        )
        cb_async = FakeAsyncCallbackHandler()
        llm_astream = llm.astream("Dummy message", config={"callbacks": [cb_async]})
        for _ in range(i):
            await llm_astream.__anext__()
        with pytest.raises(FakeListChatModelError):
            await llm_astream.__anext__()
        eval_response(cb_async, i)

        cb_sync = FakeCallbackHandler()
        llm_stream = llm.stream("Dumy message", config={"callbacks": [cb_sync]})
        for _ in range(i):
            next(llm_stream)
        with pytest.raises(FakeListChatModelError):
            next(llm_stream)
        eval_response(cb_sync, i)


async def test_astream_fallback_to_ainvoke() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithGenerate(BaseChatModel):
        @override
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
    # BaseChatModel.stream is typed to return Iterator[BaseMessageChunk].
    # When streaming is disabled, it returns Iterator[BaseMessage], so the type hint
    # is not strictly correct.
    # LangChain documents a pattern of adding BaseMessageChunks to accumulate a stream.
    # This may be better done with `reduce(operator.add, chunks)`.
    assert chunks == [_any_id_ai_message(content="hello")]  # type: ignore[comparison-overlap]

    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == [_any_id_ai_message(content="hello")]  # type: ignore[comparison-overlap]


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

        @override
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

        @override
        async def _astream(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,  # type: ignore[override]
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
    @override
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
    @override
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
    *,
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

    expected = "invoke" if disable_streaming in {"tool_calling", True} else "stream"
    assert next(model.stream([], tools=[{"type": "function"}])).content == expected
    assert (
        model.invoke(
            [], config={"callbacks": [LogStreamCallbackHandler()]}, tools=[{}]
        ).content
        == expected
    )


@pytest.mark.parametrize("disable_streaming", [True, False, "tool_calling"])
async def test_disable_streaming_async(
    *,
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

    expected = "invoke" if disable_streaming in {"tool_calling", True} else "stream"
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
    *,
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
    *,
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


class FakeChatModelStartTracer(FakeTracer):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list = []

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> Run:
        _, messages = args
        self.messages.append(messages)
        return super().on_chat_model_start(
            *args,
            **kwargs,
        )


def test_trace_images_in_openai_format() -> None:
    """Test that images are traced in OpenAI format."""
    llm = ParrotFakeChatModel()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_type": "url",
                    "url": "https://example.com/image.png",
                }
            ],
        }
    ]
    tracer = FakeChatModelStartTracer()
    response = llm.invoke(messages, config={"callbacks": [tracer]})
    assert tracer.messages == [
        [
            [
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        }
                    ]
                )
            ]
        ]
    ]
    # Test no mutation
    assert response.content == [
        {
            "type": "image",
            "source_type": "url",
            "url": "https://example.com/image.png",
        }
    ]


def test_trace_content_blocks_with_no_type_key() -> None:
    """Test that we add a ``type`` key to certain content blocks that don't have one."""
    llm = ParrotFakeChatModel()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello",
                },
                {
                    "cachePoint": {"type": "default"},
                },
            ],
        }
    ]
    tracer = FakeChatModelStartTracer()
    response = llm.invoke(messages, config={"callbacks": [tracer]})
    assert tracer.messages == [
        [
            [
                HumanMessage(
                    [
                        {
                            "type": "text",
                            "text": "Hello",
                        },
                        {
                            "type": "cachePoint",
                            "cachePoint": {"type": "default"},
                        },
                    ]
                )
            ]
        ]
    ]
    # Test no mutation
    assert response.content == [
        {
            "type": "text",
            "text": "Hello",
        },
        {
            "cachePoint": {"type": "default"},
        },
    ]


def test_extend_support_to_openai_multimodal_formats() -> None:
    """Test that chat models normalize OpenAI file and audio inputs."""
    llm = ParrotFakeChatModel()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."},
                },
                {
                    "type": "file",
                    "file": {
                        "filename": "draconomicon.pdf",
                        "file_data": "data:application/pdf;base64,<base64 string>",
                    },
                },
                {
                    "type": "file",
                    "file": {
                        "file_data": "data:application/pdf;base64,<base64 string>",
                    },
                },
                {
                    "type": "file",
                    "file": {"file_id": "<file id>"},
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": "<base64 data>", "format": "wav"},
                },
            ],
        },
    ]
    expected_content = [
        {"type": "text", "text": "Hello"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."},
        },
        {
            "type": "file",
            "source_type": "base64",
            "data": "<base64 string>",
            "mime_type": "application/pdf",
            "filename": "draconomicon.pdf",
        },
        {
            "type": "file",
            "source_type": "base64",
            "data": "<base64 string>",
            "mime_type": "application/pdf",
        },
        {
            "type": "file",
            "file": {"file_id": "<file id>"},
        },
        {
            "type": "audio",
            "source_type": "base64",
            "data": "<base64 data>",
            "mime_type": "audio/wav",
        },
    ]
    response = llm.invoke(messages)
    assert response.content == expected_content

    # Test no mutation
    assert messages[0]["content"] == [
        {"type": "text", "text": "Hello"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."},
        },
        {
            "type": "file",
            "file": {
                "filename": "draconomicon.pdf",
                "file_data": "data:application/pdf;base64,<base64 string>",
            },
        },
        {
            "type": "file",
            "file": {
                "file_data": "data:application/pdf;base64,<base64 string>",
            },
        },
        {
            "type": "file",
            "file": {"file_id": "<file id>"},
        },
        {
            "type": "input_audio",
            "input_audio": {"data": "<base64 data>", "format": "wav"},
        },
    ]


def test_normalize_messages_edge_cases() -> None:
    # Test some blocks that should pass through
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "file",
                    "file": "uri",
                },
                {
                    "type": "input_file",
                    "file_data": "uri",
                    "filename": "file-name",
                },
                {
                    "type": "input_audio",
                    "input_audio": "uri",
                },
                {
                    "type": "input_image",
                    "image_url": "uri",
                },
            ]
        )
    ]
    assert messages == _normalize_messages(messages)
