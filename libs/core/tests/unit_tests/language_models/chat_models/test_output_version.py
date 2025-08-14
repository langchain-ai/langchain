"""Test output_version functionality in BaseChatModel."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional, Union
from unittest.mock import patch

import pytest
from pydantic import ConfigDict
from typing_extensions import override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class MockChatModel(BaseChatModel):
    """Mock chat model to test output_version functionality."""

    model_config = ConfigDict(extra="allow")

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        output_version: str = "v0",
        **kwargs: Any,
    ) -> ChatResult:
        """Store the output_version that was passed."""
        self.last_output_version = output_version
        message = AIMessage(content="test response")
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        output_version: str = "v0",
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Store the output_version that was passed."""
        self.last_output_version = output_version
        yield ChatGenerationChunk(message=AIMessageChunk(content="test"))
        yield ChatGenerationChunk(message=AIMessageChunk(content=" stream"))

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,  # type: ignore[override]
        *,
        output_version: str = "v0",
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Store the output_version that was passed."""
        self.last_output_version = output_version
        yield ChatGenerationChunk(message=AIMessageChunk(content="async"))
        yield ChatGenerationChunk(message=AIMessageChunk(content=" stream"))

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


@pytest.fixture
def messages() -> list[BaseMessage]:
    return [HumanMessage("Hello")]


def test_invoke_uses_default_output_version(messages: list[BaseMessage]) -> None:
    """Test `invoke()` uses the model's default `output_version` when not specified."""
    model = MockChatModel(output_version="v1")
    model.invoke(messages)
    assert model.last_output_version == "v1"


def test_invoke_uses_provided_output_version(messages: list[BaseMessage]) -> None:
    """Test that `invoke()` uses the provided `output_version` parameter."""
    model = MockChatModel(output_version="v0")
    model.invoke(messages, output_version="v1")
    assert model.last_output_version == "v1"


def test_invoke_output_version_none_uses_default(messages: list[BaseMessage]) -> None:
    """Test that passing `output_version=None` uses the model's default."""
    model = MockChatModel(output_version="v1")
    model.invoke(messages, output_version=None)
    assert model.last_output_version == "v1"


async def test_ainvoke_uses_default_output_version(messages: list[BaseMessage]) -> None:
    """Test `ainvoke()` uses the model's default `output_version` when not specified."""
    model = MockChatModel(output_version="v1")
    await model.ainvoke(messages)
    assert model.last_output_version == "v1"


async def test_ainvoke_uses_provided_output_version(
    messages: list[BaseMessage],
) -> None:
    """Test that `ainvoke()` uses the provided `output_version` parameter."""
    model = MockChatModel(output_version="v0")
    await model.ainvoke(messages, output_version="v1")
    assert model.last_output_version == "v1"


async def test_ainvoke_output_version_none_uses_default(
    messages: list[BaseMessage],
) -> None:
    """Test that passing `output_version=None` uses the model's default."""
    model = MockChatModel(output_version="v1")
    await model.ainvoke(messages, output_version=None)
    assert model.last_output_version == "v1"


def test_stream_uses_default_output_version(messages: list[BaseMessage]) -> None:
    """Test `stream()` uses the model's default `output_version` when not specified."""
    model = MockChatModel(output_version="v1")
    list(model.stream(messages))
    assert model.last_output_version == "v1"


def test_stream_uses_provided_output_version(messages: list[BaseMessage]) -> None:
    """Test that `stream()` uses the provided `output_version` parameter."""
    model = MockChatModel(output_version="v1")
    list(model.stream(messages, output_version="v2"))
    assert model.last_output_version == "v2"


def test_stream_output_version_none_uses_default(messages: list[BaseMessage]) -> None:
    """Test that passing `output_version=None` uses the model's default."""
    model = MockChatModel(output_version="v1")
    list(model.stream(messages, output_version=None))
    assert model.last_output_version == "v1"


async def test_astream_uses_default_output_version(messages: list[BaseMessage]) -> None:
    """Test `astream()` uses the model's default `output_version` when not specified."""
    model = MockChatModel(output_version="v1")
    async for _ in model.astream(messages):
        pass
    assert model.last_output_version == "v1"


async def test_astream_uses_provided_output_version(
    messages: list[BaseMessage],
) -> None:
    """Test that `astream()` uses the provided `output_version` parameter."""
    model = MockChatModel(output_version="v1")
    async for _ in model.astream(messages, output_version="v0"):
        pass
    assert model.last_output_version == "v0"


async def test_astream_output_version_none_uses_default(
    messages: list[BaseMessage],
) -> None:
    """Test that passing `output_version=None` uses the model's default."""
    model = MockChatModel(output_version="v1")
    async for _ in model.astream(messages, output_version=None):
        pass
    assert model.last_output_version == "v1"


def test_stream_fallback_to_invoke_passes_output_version(
    messages: list[BaseMessage],
) -> None:
    """Test that `stream()` fallback to invoke passes the `output_version` correctly."""

    class NoStreamModel(BaseChatModel):
        model_config = ConfigDict(extra="allow")

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            *,
            output_version: str = "v0",
            **kwargs: Any,
        ) -> ChatResult:
            self.last_output_version = output_version
            message = AIMessage(content="test response")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        @property
        def _llm_type(self) -> str:
            return "no-stream-model"

    model = NoStreamModel(output_version="v1")
    # Stream should fallback to invoke and pass the output_version
    list(model.stream(messages, output_version="v2"))
    assert model.last_output_version == "v2"


async def test_astream_fallback_to_ainvoke_passes_output_version(
    messages: list[BaseMessage],
) -> None:
    """Test `astream()` fallback to ainvoke passes the `output_version` correctly."""

    class NoStreamModel(BaseChatModel):
        model_config = ConfigDict(extra="allow")

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            *,
            output_version: str = "v0",
            **kwargs: Any,
        ) -> ChatResult:
            self.last_output_version = output_version
            message = AIMessage(content="test response")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        @property
        def _llm_type(self) -> str:
            return "no-stream-model"

    model = NoStreamModel(output_version="v1")
    # Astream should fallback to ainvoke and pass the output_version
    async for _ in model.astream(messages, output_version="v2"):
        pass
    assert model.last_output_version == "v2"


def test_generate_prompt_passes_output_version_to_internal_methods(
    messages: list[BaseMessage],
) -> None:
    """Test `generate_prompt()` passes `output_version` to internal `_generate()`."""
    model = MockChatModel(output_version="v1")

    # Mock the _generate method to verify it receives the output_version
    with patch.object(model, "_generate", wraps=model._generate) as mock_generate:
        model.invoke(messages, output_version="v2")
        mock_generate.assert_called_once()
        # Verify that _generate was called with output_version="v2"
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs.get("output_version") == "v2"


async def test_agenerate_prompt_passes_output_version_to_internal_methods(
    messages: list[BaseMessage],
) -> None:
    """Test `agenerate_prompt()` passes output_version to internal `_agenerate()`."""
    model = MockChatModel(output_version="v1")

    # Mock the _agenerate method to verify it receives the output_version
    with patch.object(model, "_agenerate", wraps=model._agenerate) as mock_agenerate:
        await model.ainvoke(messages, output_version="v2")
        mock_agenerate.assert_called_once()
        # Verify that _agenerate was called with output_version="v2"
        call_kwargs = mock_agenerate.call_args.kwargs
        assert call_kwargs.get("output_version") == "v2"


def test_different_output_versions() -> None:
    """Test that different `output_version` values are handled correctly."""
    messages = [HumanMessage(content="Hello")]
    model = MockChatModel(output_version="v0")

    # Test with various output version strings
    test_versions = ["v0", "v1", "v2", "beta", "experimental", "1.0", "2025-01-01"]

    for version in test_versions:
        model.invoke(messages, output_version=version)
        assert model.last_output_version == version


def test_output_version_is_keyword_only() -> None:
    """Test that `output_version` parameter is keyword-only in public methods."""
    messages = [HumanMessage(content="Hello")]
    model = MockChatModel()

    # These should work (keyword argument)
    model.invoke(messages, output_version="v1")
    list(model.stream(messages, output_version="v1"))

    # These should fail if output_version were positional (but they don't because
    # it's after the * in the signature, making it keyword-only)
    with pytest.raises(TypeError):
        model.invoke(messages, None, "v1")  # type: ignore[arg-type,misc]

    with pytest.raises(TypeError):
        list(model.stream(messages, None, "v1"))  # type: ignore[arg-type,misc]


async def test_async_output_version_is_keyword_only() -> None:
    """Test that `output_version` parameter is keyword-only in async public methods."""
    messages = [HumanMessage(content="Hello")]
    model = MockChatModel()

    # These should work (keyword argument)
    await model.ainvoke(messages, output_version="v1")
    async for _ in model.astream(messages, output_version="v1"):
        pass

    # These should fail if output_version were positional
    with pytest.raises(TypeError):
        await model.ainvoke(messages, None, "v1")  # type: ignore[arg-type,misc]

    with pytest.raises(TypeError):
        async for _ in model.astream(messages, None, "v1"):  # type: ignore[arg-type,misc]
            pass


def test_output_version_inheritance() -> None:
    """Test that subclasses properly inherit `output_version` functionality."""

    class CustomChatModel(BaseChatModel):
        model_config = ConfigDict(extra="allow")

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.received_versions: list[str] = []

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            *,
            output_version: str = "v0",
            **kwargs: Any,
        ) -> ChatResult:
            self.received_versions.append(output_version)
            message = AIMessage(content="response")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        @property
        def _llm_type(self) -> str:
            return "custom-model"

    messages = [HumanMessage(content="Hello")]
    model = CustomChatModel(output_version="default_v1")

    # Test that default is used
    model.invoke(messages)
    assert model.received_versions[-1] == "default_v1"

    # Test that override is used
    model.invoke(messages, output_version="override_v2")
    assert model.received_versions[-1] == "override_v2"


def test_internal_output_version_parameter_in_signature() -> None:
    """Test that internal methods have `output_version` in their signatures."""
    import inspect

    model = MockChatModel()

    # Check that the internal methods have output_version parameters
    generate_sig = inspect.signature(model._generate)
    assert "output_version" in generate_sig.parameters
    assert generate_sig.parameters["output_version"].default == "v0"

    agenerate_sig = inspect.signature(model._agenerate)
    assert "output_version" in agenerate_sig.parameters
    assert agenerate_sig.parameters["output_version"].default == "v0"

    stream_sig = inspect.signature(model._stream)
    assert "output_version" in stream_sig.parameters
    assert stream_sig.parameters["output_version"].default == "v0"

    astream_sig = inspect.signature(model._astream)
    assert "output_version" in astream_sig.parameters
    assert astream_sig.parameters["output_version"].default == "v0"


@pytest.mark.parametrize(
    ("model_version", "expected"),
    [
        ("v0", "v0"),
        ("v1", "v1"),
        ("responses/v1", "responses/v1"),
    ],
)
def test_output_version_stored_in_additional_kwargs_invoke(
    messages: list[BaseMessage], model_version: str, expected: str
) -> None:
    """Test that output_version is stored in message additional_kwargs for invoke."""
    model = MockChatModel(output_version=model_version)
    response = model.invoke(messages)

    assert "output_version" in response.additional_kwargs
    assert response.additional_kwargs["output_version"] == expected


@pytest.mark.parametrize(
    ("model_version", "override_version", "expected"),
    [
        ("v0", None, "v0"),
        ("v1", None, "v1"),
        ("v0", "v2", "v2"),
        ("v1", "v0", "v0"),
    ],
)
async def test_output_version_ainvoke_with_override(
    messages: list[BaseMessage],
    model_version: str,
    override_version: str,
    expected: str,
) -> None:
    """Test ainvoke with output_version override."""
    model = MockChatModel(output_version=model_version)
    response = await model.ainvoke(messages, output_version=override_version)

    assert "output_version" in response.additional_kwargs
    assert response.additional_kwargs["output_version"] == expected


@pytest.mark.parametrize(
    ("model_version", "override_version", "expected"),
    [
        ("v0", None, "v0"),
        ("v1", None, "v1"),
        ("v0", "v2", "v2"),
    ],
)
def test_output_version_stored_in_stream_chunks(
    messages: list[BaseMessage],
    model_version: str,
    override_version: str,
    expected: str,
) -> None:
    """Test that output_version is stored in streaming chunk additional_kwargs."""
    model = MockChatModel(output_version=model_version)
    chunks = list(model.stream(messages, output_version=override_version))

    for chunk in chunks:
        assert "output_version" in chunk.additional_kwargs
        assert chunk.additional_kwargs["output_version"] == expected


@pytest.mark.parametrize(
    ("model_version", "override_version", "expected"),
    [
        ("v0", None, "v0"),
        ("v1", None, "v1"),
        ("v0", "v2", "v2"),
    ],
)
async def test_output_version_stored_in_astream_chunks(
    messages: list[BaseMessage],
    model_version: str,
    override_version: str,
    expected: str,
) -> None:
    """Test that output_version is stored in async streaming chunk additional_kwargs."""
    model = MockChatModel(output_version=model_version)
    chunks = [
        chunk
        async for chunk in model.astream(messages, output_version=override_version)
    ]

    for chunk in chunks:
        assert "output_version" in chunk.additional_kwargs
        assert chunk.additional_kwargs["output_version"] == expected


@pytest.mark.parametrize("version", ["v0", "v1", "v2", "beta", "responses/v1"])
def test_output_version_preserved_through_serialization(
    messages: list[BaseMessage], version: str
) -> None:
    """Test that output_version in additional_kwargs persists through serialization."""
    import json

    model = MockChatModel(output_version="v0")
    response = model.invoke(messages, output_version=version)
    assert response.additional_kwargs["output_version"] == version

    # Verify serialization preserves version
    message_dict = {"additional_kwargs": response.additional_kwargs}
    serialized = json.dumps(message_dict)
    deserialized = json.loads(serialized)
    assert deserialized["additional_kwargs"]["output_version"] == version


@pytest.mark.parametrize(
    ("output_version", "content_type"),
    [
        ("v0", str),
        ("v1", list),
    ],
)
def test_output_version_with_different_content_formats(
    messages: list[BaseMessage], output_version: str, content_type: type
) -> None:
    """Test output_version storage works with different content formats."""

    class CustomChatModel(BaseChatModel):
        model_config = ConfigDict(extra="allow")

        def _generate(
            self,
            messages: list[BaseMessage],  # noqa: ARG002
            stop: Optional[list[str]] = None,  # noqa: ARG002
            run_manager: Optional[CallbackManagerForLLMRun] = None,  # noqa: ARG002
            *,
            output_version: str = "v0",
            **kwargs: Any,  # noqa: ARG002
        ) -> ChatResult:
            if output_version == "v0":
                content: Union[str, list[dict[str, Any]]] = "test response"
            else:
                content = [{"type": "text", "text": "test response"}]
            message = AIMessage(content=content)  # type: ignore[arg-type]
            return ChatResult(generations=[ChatGeneration(message=message)])

        @property
        def _llm_type(self) -> str:
            return "custom-test-model"

    model = CustomChatModel()
    response = model.invoke(messages, output_version=output_version)

    assert response.additional_kwargs["output_version"] == output_version
    assert isinstance(response.content, content_type)


def test_output_version_preserves_existing_additional_kwargs(
    messages: list[BaseMessage],
) -> None:
    """Test that output_version doesn't overwrite existing additional_kwargs."""

    class ModelWithExistingKwargs(BaseChatModel):
        model_config = ConfigDict(extra="allow")

        def _generate(
            self,
            messages: list[BaseMessage],  # noqa: ARG002
            stop: Optional[list[str]] = None,  # noqa: ARG002
            run_manager: Optional[CallbackManagerForLLMRun] = None,  # noqa: ARG002
            *,
            output_version: str = "v0",  # noqa: ARG002
            **kwargs: Any,  # noqa: ARG002
        ) -> ChatResult:
            message = AIMessage(
                content="test response",
                additional_kwargs={"model": "test-model", "temperature": 0.7},
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

        @property
        def _llm_type(self) -> str:
            return "kwargs-test-model"

    model = ModelWithExistingKwargs(output_version="v1")
    response = model.invoke(messages)

    # Verify output_version was added and existing kwargs preserved
    assert response.additional_kwargs["output_version"] == "v1"
    assert response.additional_kwargs["model"] == "test-model"
    assert response.additional_kwargs["temperature"] == 0.7
    assert len(response.additional_kwargs) == 3
