"""Test base chat model."""

import uuid
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Literal

import pytest
from typing_extensions import override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import (
    BaseChatModel,
    FakeListChatModel,
    ParrotFakeChatModel,
)
from langchain_core.language_models._utils import _normalize_messages
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModelError,
    GenericFakeChatModel,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
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


def _content_blocks_equal_ignore_id(
    actual: str | list[Any], expected: str | list[Any]
) -> bool:
    """Compare content blocks, ignoring auto-generated `id` fields.

    Args:
        actual: Actual content from response (string or list of content blocks).
        expected: Expected content to compare against (string or list of blocks).

    Returns:
        True if content matches (excluding `id` fields), False otherwise.

    """
    if isinstance(actual, str) or isinstance(expected, str):
        return actual == expected

    if len(actual) != len(expected):
        return False
    for actual_block, expected_block in zip(actual, expected, strict=False):
        actual_without_id = (
            {k: v for k, v in actual_block.items() if k != "id"}
            if isinstance(actual_block, dict) and "id" in actual_block
            else actual_block
        )

        if actual_without_id != expected_block:
            return False

    return True


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
    """Test `astream()` uses appropriate implementation."""

    class ModelWithGenerate(BaseChatModel):
        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
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
    assert chunks == [_any_id_ai_message(content="hello")]

    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == [_any_id_ai_message(content="hello")]


async def test_astream_implementation_fallback_to_stream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithSyncStream(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Top Level call."""
            raise NotImplementedError

        @override
        def _stream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> Iterator[ChatGenerationChunk]:
            """Stream the output of the model."""
            yield ChatGenerationChunk(message=AIMessageChunk(content="a"))
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="b", chunk_position="last")
            )

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithSyncStream()
    chunks = list(model.stream("anything"))
    assert chunks == [
        _any_id_ai_message_chunk(
            content="a",
        ),
        _any_id_ai_message_chunk(content="b", chunk_position="last"),
    ]
    assert len({chunk.id for chunk in chunks}) == 1
    assert type(model)._astream == BaseChatModel._astream
    astream_chunks = [chunk async for chunk in model.astream("anything")]
    assert astream_chunks == [
        _any_id_ai_message_chunk(
            content="a",
        ),
        _any_id_ai_message_chunk(content="b", chunk_position="last"),
    ]
    assert len({chunk.id for chunk in astream_chunks}) == 1


async def test_astream_implementation_uses_astream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithAsyncStream(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Top Level call."""
            raise NotImplementedError

        @override
        async def _astream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,  # type: ignore[override]
            **kwargs: Any,
        ) -> AsyncIterator[ChatGenerationChunk]:
            """Stream the output of the model."""
            yield ChatGenerationChunk(message=AIMessageChunk(content="a"))
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="b", chunk_position="last")
            )

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithAsyncStream()
    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == [
        _any_id_ai_message_chunk(
            content="a",
        ),
        _any_id_ai_message_chunk(content="b", chunk_position="last"),
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
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
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
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="stream"))


@pytest.mark.parametrize("disable_streaming", [True, False, "tool_calling"])
def test_disable_streaming(
    *,
    disable_streaming: bool | Literal["tool_calling"],
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
    disable_streaming: bool | Literal["tool_calling"],
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
    disable_streaming: bool | Literal["tool_calling"],
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
    disable_streaming: bool | Literal["tool_calling"],
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
    """Test that images are traced in OpenAI Chat Completions format."""
    llm = ParrotFakeChatModel()
    messages = [
        {
            "role": "user",
            # v0 format
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
    llm.invoke(messages, config={"callbacks": [tracer]})
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


def test_trace_pdfs() -> None:
    # For backward compat
    llm = ParrotFakeChatModel()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "mime_type": "application/pdf",
                    "base64": "<base64 string>",
                }
            ],
        }
    ]
    tracer = FakeChatModelStartTracer()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        llm.invoke(messages, config={"callbacks": [tracer]})

    assert tracer.messages == [
        [
            [
                HumanMessage(
                    content=[
                        {
                            "type": "file",
                            "mime_type": "application/pdf",
                            "source_type": "base64",
                            "data": "<base64 string>",
                        }
                    ]
                )
            ]
        ]
    ]


def test_content_block_transformation_v0_to_v1_image() -> None:
    """Test that v0 format image content blocks are transformed to v1 format."""
    # Create a message with v0 format image content
    image_message = AIMessage(
        content=[
            {
                "type": "image",
                "source_type": "url",
                "url": "https://example.com/image.png",
            }
        ]
    )

    llm = GenericFakeChatModel(messages=iter([image_message]), output_version="v1")
    response = llm.invoke("test")

    # With v1 output_version, .content should be transformed
    # Check structure, ignoring auto-generated IDs
    assert len(response.content) == 1
    content_block = response.content[0]
    if isinstance(content_block, dict) and "id" in content_block:
        # Remove auto-generated id for comparison
        content_without_id = {k: v for k, v in content_block.items() if k != "id"}
        expected_content = {
            "type": "image",
            "url": "https://example.com/image.png",
        }
        assert content_without_id == expected_content
    else:
        assert content_block == {
            "type": "image",
            "url": "https://example.com/image.png",
        }


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_trace_content_blocks_with_no_type_key(output_version: str) -> None:
    """Test behavior of content blocks that don't have a `type` key.

    Only for blocks with one key, in which case, the name of the key is used as `type`.

    """
    llm = ParrotFakeChatModel(output_version=output_version)
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

    if output_version == "v0":
        assert response.content == [
            {
                "type": "text",
                "text": "Hello",
            },
            {
                "cachePoint": {"type": "default"},
            },
        ]
    else:
        assert response.content == [
            {
                "type": "text",
                "text": "Hello",
            },
            {
                "type": "non_standard",
                "value": {
                    "cachePoint": {"type": "default"},
                },
            },
        ]

    assert response.content_blocks == [
        {
            "type": "text",
            "text": "Hello",
        },
        {
            "type": "non_standard",
            "value": {
                "cachePoint": {"type": "default"},
            },
        },
    ]


def test_extend_support_to_openai_multimodal_formats() -> None:
    """Test normalizing OpenAI audio, image, and file inputs to v1."""
    # Audio and file only (chat model default)
    messages = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},
            {  # audio-base64
                "type": "input_audio",
                "input_audio": {
                    "format": "wav",
                    "data": "<base64 string>",
                },
            },
            {  # file-base64
                "type": "file",
                "file": {
                    "filename": "draconomicon.pdf",
                    "file_data": "data:application/pdf;base64,<base64 string>",
                },
            },
            {  # file-id
                "type": "file",
                "file": {"file_id": "<file id>"},
            },
        ]
    )

    expected_content_messages = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},  # TextContentBlock
            {  # AudioContentBlock
                "type": "audio",
                "base64": "<base64 string>",
                "mime_type": "audio/wav",
            },
            {  # FileContentBlock
                "type": "file",
                "base64": "<base64 string>",
                "mime_type": "application/pdf",
                "extras": {"filename": "draconomicon.pdf"},
            },
            {  # ...
                "type": "file",
                "file_id": "<file id>",
            },
        ]
    )

    normalized_content = _normalize_messages([messages])

    # Check structure, ignoring auto-generated IDs
    assert len(normalized_content) == 1
    normalized_message = normalized_content[0]
    assert len(normalized_message.content) == len(expected_content_messages.content)

    assert _content_blocks_equal_ignore_id(
        normalized_message.content, expected_content_messages.content
    )

    messages = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},
            {  # image-url
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
            {  # image-base64
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."},
            },
            {  # audio-base64
                "type": "input_audio",
                "input_audio": {
                    "format": "wav",
                    "data": "<base64 string>",
                },
            },
            {  # file-base64
                "type": "file",
                "file": {
                    "filename": "draconomicon.pdf",
                    "file_data": "data:application/pdf;base64,<base64 string>",
                },
            },
            {  # file-id
                "type": "file",
                "file": {"file_id": "<file id>"},
            },
        ]
    )

    expected_content_messages = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},  # TextContentBlock
            {  # image-url passes through
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
            {  # image-url passes through with inline data
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."},
            },
            {  # AudioContentBlock
                "type": "audio",
                "base64": "<base64 string>",
                "mime_type": "audio/wav",
            },
            {  # FileContentBlock
                "type": "file",
                "base64": "<base64 string>",
                "mime_type": "application/pdf",
                "extras": {"filename": "draconomicon.pdf"},
            },
            {  # ...
                "type": "file",
                "file_id": "<file id>",
            },
        ]
    )

    normalized_content = _normalize_messages([messages])

    # Check structure, ignoring auto-generated IDs
    assert len(normalized_content) == 1
    normalized_message = normalized_content[0]
    assert len(normalized_message.content) == len(expected_content_messages.content)

    assert _content_blocks_equal_ignore_id(
        normalized_message.content, expected_content_messages.content
    )


def test_normalize_messages_edge_cases() -> None:
    # Test behavior of malformed/unrecognized content blocks

    messages = [
        HumanMessage(
            content=[
                {
                    "type": "input_image",  # Responses API type; not handled
                    "image_url": "uri",
                },
                {
                    # Standard OpenAI Chat Completions type but malformed structure
                    "type": "input_audio",
                    "input_audio": "uri",  # Should be nested in `audio`
                },
                {
                    "type": "file",
                    "file": "uri",  # `file` should be a dict for Chat Completions
                },
                {
                    "type": "input_file",  # Responses API type; not handled
                    "file_data": "uri",
                    "filename": "file-name",
                },
            ]
        )
    ]

    assert messages == _normalize_messages(messages)


def test_normalize_messages_v1_content_blocks_unchanged() -> None:
    """Test passing v1 content blocks to `_normalize_messages()` leaves unchanged."""
    input_messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Hello world",
                },
                {
                    "type": "image",
                    "url": "https://example.com/image.png",
                    "mime_type": "image/png",
                },
                {
                    "type": "audio",
                    "base64": "base64encodedaudiodata",
                    "mime_type": "audio/wav",
                },
                {
                    "type": "file",
                    "id": "file_123",
                },
                {
                    "type": "reasoning",
                    "reasoning": "Let me think about this...",
                },
            ]
        )
    ]

    result = _normalize_messages(input_messages)

    # Verify the result is identical to the input (message should not be copied)
    assert len(result) == 1
    assert result[0] is input_messages[0]
    assert result[0].content == input_messages[0].content


def test_output_version_invoke(monkeypatch: Any) -> None:
    messages = [AIMessage("hello")]

    llm = GenericFakeChatModel(messages=iter(messages), output_version="v1")
    response = llm.invoke("hello")
    assert response.content == [{"type": "text", "text": "hello"}]
    assert response.response_metadata["output_version"] == "v1"

    llm = GenericFakeChatModel(messages=iter(messages))
    response = llm.invoke("hello")
    assert response.content == "hello"

    monkeypatch.setenv("LC_OUTPUT_VERSION", "v1")
    llm = GenericFakeChatModel(messages=iter(messages))
    response = llm.invoke("hello")
    assert response.content == [{"type": "text", "text": "hello"}]
    assert response.response_metadata["output_version"] == "v1"


# -- v1 output version tests --


async def test_output_version_ainvoke(monkeypatch: Any) -> None:
    messages = [AIMessage("hello")]

    # v0
    llm = GenericFakeChatModel(messages=iter(messages))
    response = await llm.ainvoke("hello")
    assert response.content == "hello"

    # v1
    llm = GenericFakeChatModel(messages=iter(messages), output_version="v1")
    response = await llm.ainvoke("hello")
    assert response.content == [{"type": "text", "text": "hello"}]
    assert response.response_metadata["output_version"] == "v1"

    # v1 from env var
    monkeypatch.setenv("LC_OUTPUT_VERSION", "v1")
    llm = GenericFakeChatModel(messages=iter(messages))
    response = await llm.ainvoke("hello")
    assert response.content == [{"type": "text", "text": "hello"}]
    assert response.response_metadata["output_version"] == "v1"


def test_output_version_stream(monkeypatch: Any) -> None:
    messages = [AIMessage("foo bar")]

    # v0
    llm = GenericFakeChatModel(messages=iter(messages))
    full = None
    for chunk in llm.stream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, str)
        assert chunk.content
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.content == "foo bar"

    # v1
    llm = GenericFakeChatModel(messages=iter(messages), output_version="v1")
    full_v1: BaseMessageChunk | None = None
    for chunk in llm.stream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, list)
        assert len(chunk.content) == 1
        block = chunk.content[0]
        assert isinstance(block, dict)
        assert block["type"] == "text"
        assert block["text"]
        full_v1 = chunk if full_v1 is None else full_v1 + chunk
    assert isinstance(full_v1, AIMessageChunk)
    assert full_v1.response_metadata["output_version"] == "v1"

    # v1 from env var
    monkeypatch.setenv("LC_OUTPUT_VERSION", "v1")
    llm = GenericFakeChatModel(messages=iter(messages))
    full_env = None
    for chunk in llm.stream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, list)
        assert len(chunk.content) == 1
        block = chunk.content[0]
        assert isinstance(block, dict)
        assert block["type"] == "text"
        assert block["text"]
        full_env = chunk if full_env is None else full_env + chunk
    assert isinstance(full_env, AIMessageChunk)
    assert full_env.response_metadata["output_version"] == "v1"


async def test_output_version_astream(monkeypatch: Any) -> None:
    messages = [AIMessage("foo bar")]

    # v0
    llm = GenericFakeChatModel(messages=iter(messages))
    full = None
    async for chunk in llm.astream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, str)
        assert chunk.content
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.content == "foo bar"

    # v1
    llm = GenericFakeChatModel(messages=iter(messages), output_version="v1")
    full_v1: BaseMessageChunk | None = None
    async for chunk in llm.astream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, list)
        assert len(chunk.content) == 1
        block = chunk.content[0]
        assert isinstance(block, dict)
        assert block["type"] == "text"
        assert block["text"]
        full_v1 = chunk if full_v1 is None else full_v1 + chunk
    assert isinstance(full_v1, AIMessageChunk)
    assert full_v1.response_metadata["output_version"] == "v1"

    # v1 from env var
    monkeypatch.setenv("LC_OUTPUT_VERSION", "v1")
    llm = GenericFakeChatModel(messages=iter(messages))
    full_env = None
    async for chunk in llm.astream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, list)
        assert len(chunk.content) == 1
        block = chunk.content[0]
        assert isinstance(block, dict)
        assert block["type"] == "text"
        assert block["text"]
        full_env = chunk if full_env is None else full_env + chunk
    assert isinstance(full_env, AIMessageChunk)
    assert full_env.response_metadata["output_version"] == "v1"
    assert messages == _normalize_messages(messages)


def test_get_ls_params() -> None:
    class LSParamsModel(BaseChatModel):
        model: str = "foo"
        temperature: float = 0.1
        max_tokens: int = 1024

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            raise NotImplementedError

        @override
        def _stream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> Iterator[ChatGenerationChunk]:
            raise NotImplementedError

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    llm = LSParamsModel()

    # Test standard tracing params
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "lsparamsmodel",
        "ls_model_type": "chat",
        "ls_model_name": "foo",
        "ls_temperature": 0.1,
        "ls_max_tokens": 1024,
    }

    ls_params = llm._get_ls_params(model="bar")
    assert ls_params["ls_model_name"] == "bar"

    ls_params = llm._get_ls_params(temperature=0.2)
    assert ls_params["ls_temperature"] == 0.2

    ls_params = llm._get_ls_params(max_tokens=2048)
    assert ls_params["ls_max_tokens"] == 2048

    ls_params = llm._get_ls_params(stop=["stop"])
    assert ls_params["ls_stop"] == ["stop"]
