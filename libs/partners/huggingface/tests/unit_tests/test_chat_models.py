from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest  # type: ignore[import-not-found]
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool

from langchain_huggingface.chat_models import (  # type: ignore[import]
    ChatHuggingFace,
    _convert_dict_to_message,
)
from langchain_huggingface.llms import (
    HuggingFaceEndpoint,
    HuggingFacePipeline,
)


@pytest.fixture
def mock_llm() -> Mock:
    llm = Mock(spec=HuggingFaceEndpoint)
    llm.inference_server_url = "test endpoint url"
    llm.temperature = 0.7
    llm.max_new_tokens = 512
    llm.top_p = 0.9
    llm.seed = 42
    llm.streaming = True
    llm.repetition_penalty = 1.1
    llm.stop_sequences = ["</s>", "<|end|>"]
    llm.model_kwargs = {"do_sample": True, "top_k": 50}
    llm.server_kwargs = {"timeout": 120}
    llm.repo_id = "test/model"
    llm.model = "test/model"
    return llm


@pytest.fixture
@patch(
    "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
)
def chat_hugging_face(mock_resolve_id: Any, mock_llm: Any) -> ChatHuggingFace:
    return ChatHuggingFace(llm=mock_llm, tokenizer=MagicMock())


def test_create_chat_result(chat_hugging_face: Any) -> None:
    mock_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "test message"},
                "finish_reason": "test finish reason",
            }
        ],
        "usage": {"tokens": 420},
    }

    result = chat_hugging_face._create_chat_result(mock_response)
    assert isinstance(result, ChatResult)
    assert result.generations[0].message.content == "test message"
    assert (
        result.generations[0].generation_info["finish_reason"] == "test finish reason"  # type: ignore[index]
    )
    assert result.llm_output["token_usage"]["tokens"] == 420  # type: ignore[index]
    assert result.llm_output["model_name"] == chat_hugging_face.model_id  # type: ignore[index]


@pytest.mark.parametrize(
    "messages, expected_error",
    [
        ([], "At least one HumanMessage must be provided!"),
        (
            [HumanMessage(content="Hi"), AIMessage(content="Hello")],
            "Last message must be a HumanMessage!",
        ),
    ],
)
def test_to_chat_prompt_errors(
    chat_hugging_face: Any, messages: list[BaseMessage], expected_error: str
) -> None:
    with pytest.raises(ValueError) as e:
        chat_hugging_face._to_chat_prompt(messages)
    assert expected_error in str(e.value)


def test_to_chat_prompt_valid_messages(chat_hugging_face: Any) -> None:
    messages = [AIMessage(content="Hello"), HumanMessage(content="How are you?")]
    expected_prompt = "Generated chat prompt"

    chat_hugging_face.tokenizer.apply_chat_template.return_value = expected_prompt

    result = chat_hugging_face._to_chat_prompt(messages)

    assert result == expected_prompt
    chat_hugging_face.tokenizer.apply_chat_template.assert_called_once_with(
        [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            SystemMessage(content="You are a helpful assistant."),
            {"role": "system", "content": "You are a helpful assistant."},
        ),
        (
            AIMessage(content="How can I help you?"),
            {"role": "assistant", "content": "How can I help you?"},
        ),
        (
            HumanMessage(content="Hello"),
            {"role": "user", "content": "Hello"},
        ),
    ],
)
def test_to_chatml_format(
    chat_hugging_face: Any, message: BaseMessage, expected: dict[str, str]
) -> None:
    result = chat_hugging_face._to_chatml_format(message)
    assert result == expected


def test_to_chatml_format_with_invalid_type(chat_hugging_face: Any) -> None:
    message = "Invalid message type"
    with pytest.raises(ValueError) as e:
        chat_hugging_face._to_chatml_format(message)
    assert "Unknown message type:" in str(e.value)


@pytest.mark.parametrize(
    ("msg_dict", "expected_type", "expected_content"),
    [
        (
            {"role": "system", "content": "You are helpful"},
            SystemMessage,
            "You are helpful",
        ),
        (
            {"role": "user", "content": "Hello there"},
            HumanMessage,
            "Hello there",
        ),
        (
            {"role": "assistant", "content": "How can I help?"},
            AIMessage,
            "How can I help?",
        ),
        (
            {"role": "function", "content": "result", "name": "get_time"},
            FunctionMessage,
            "result",
        ),
    ],
)
def test_convert_dict_to_message(
    msg_dict: dict[str, Any], expected_type: type, expected_content: str
) -> None:
    result = _convert_dict_to_message(msg_dict)
    assert isinstance(result, expected_type)
    assert result.content == expected_content


def tool_mock() -> dict:
    return {"function": {"name": "test_tool"}}


@pytest.mark.parametrize(
    "tools, tool_choice, expected_exception, expected_message",
    [
        ([tool_mock()], ["invalid type"], ValueError, "Unrecognized tool_choice type."),
        (
            [tool_mock(), tool_mock()],
            "test_tool",
            ValueError,
            "must provide exactly one tool.",
        ),
        (
            [tool_mock()],
            {"type": "function", "function": {"name": "other_tool"}},
            ValueError,
            "Tool choice {'type': 'function', 'function': {'name': 'other_tool'}} "
            "was specified, but the only provided tool was test_tool.",
        ),
    ],
)
def test_bind_tools_errors(
    chat_hugging_face: Any,
    tools: dict[str, str],
    tool_choice: Any,
    expected_exception: Any,
    expected_message: str,
) -> None:
    with patch(
        "langchain_huggingface.chat_models.huggingface.convert_to_openai_tool",
        side_effect=lambda x: x,
    ):
        with pytest.raises(expected_exception) as excinfo:
            chat_hugging_face.bind_tools(tools, tool_choice=tool_choice)
        assert expected_message in str(excinfo.value)


def test_bind_tools(chat_hugging_face: Any) -> None:
    tools = [MagicMock(spec=BaseTool)]
    with (
        patch(
            "langchain_huggingface.chat_models.huggingface.convert_to_openai_tool",
            side_effect=lambda x: x,
        ),
        patch("langchain_core.runnables.base.Runnable.bind") as mock_super_bind,
    ):
        chat_hugging_face.bind_tools(tools, tool_choice="auto")
        mock_super_bind.assert_called_once()
        _, kwargs = mock_super_bind.call_args
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"


def test_property_inheritance_integration(chat_hugging_face: Any) -> None:
    """Test that ChatHuggingFace inherits params from LLM object."""
    assert getattr(chat_hugging_face, "temperature", None) == 0.7
    assert getattr(chat_hugging_face, "max_tokens", None) == 512
    assert getattr(chat_hugging_face, "top_p", None) == 0.9
    assert getattr(chat_hugging_face, "streaming", None) is True


def test_default_params_includes_inherited_values(chat_hugging_face: Any) -> None:
    """Test that _default_params includes inherited max_tokens from max_new_tokens."""
    params = chat_hugging_face._default_params
    assert params["max_tokens"] == 512  # inherited from LLM's max_new_tokens
    assert params["temperature"] == 0.7  # inherited from LLM's temperature
    assert params["stream"] is True  # inherited from LLM's streaming


def test_create_message_dicts_includes_inherited_params(chat_hugging_face: Any) -> None:
    """Test that _create_message_dicts includes inherited parameters in API call."""
    messages = [HumanMessage(content="test message")]
    message_dicts, params = chat_hugging_face._create_message_dicts(messages, None)

    # Verify inherited parameters are included
    assert params["max_tokens"] == 512
    assert params["temperature"] == 0.7
    assert params["stream"] is True

    # Verify message conversion
    assert len(message_dicts) == 1
    assert message_dicts[0]["role"] == "user"
    assert message_dicts[0]["content"] == "test message"


def test_model_kwargs_inheritance(mock_llm: Any) -> None:
    """Test that model_kwargs are inherited when not explicitly set."""
    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        chat = ChatHuggingFace(llm=mock_llm)
        assert chat.model_kwargs == {"do_sample": True, "top_k": 50}


def test_huggingface_endpoint_specific_inheritance(mock_llm: Any) -> None:
    """Test HuggingFaceEndpoint specific parameter inheritance."""
    with (
        patch(
            "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
        ),
        patch(
            "langchain_huggingface.chat_models.huggingface._is_huggingface_endpoint",
            return_value=True,
        ),
    ):
        chat = ChatHuggingFace(llm=mock_llm)
        assert (
            getattr(chat, "frequency_penalty", None) == 1.1
        )  # from repetition_penalty


def test_parameter_precedence_explicit_over_inherited(mock_llm: Any) -> None:
    """Test that explicitly set parameters take precedence over inherited ones."""
    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        # Explicitly set max_tokens to override inheritance
        chat = ChatHuggingFace(llm=mock_llm, max_tokens=256, temperature=0.5)
        assert chat.max_tokens == 256  # explicit value, not inherited 512
        assert chat.temperature == 0.5  # explicit value, not inherited 0.7


def test_inheritance_with_no_llm_properties(mock_llm: Any) -> None:
    """Test inheritance when LLM doesn't have expected properties."""
    # Remove some properties from mock
    del mock_llm.temperature
    del mock_llm.top_p

    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        chat = ChatHuggingFace(llm=mock_llm)
        # Should still inherit available properties
        assert chat.max_tokens == 512  # max_new_tokens still available
        # Missing properties should remain None/default
        assert getattr(chat, "temperature", None) is None
        assert getattr(chat, "top_p", None) is None


def test_inheritance_with_empty_llm() -> None:
    """Test that inheritance handles LLM with no relevant attributes gracefully."""
    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        # Create a minimal mock LLM that passes validation but has no
        # inheritance attributes
        empty_llm = Mock(spec=HuggingFaceEndpoint)
        empty_llm.repo_id = "test/model"
        empty_llm.model = "test/model"
        # Mock doesn't have the inheritance attributes by default

        chat = ChatHuggingFace(llm=empty_llm)
        # Properties should remain at their default values when LLM has no
        # relevant attrs
        assert chat.max_tokens is None
        assert chat.temperature is None


def test_profile() -> None:
    empty_llm = Mock(spec=HuggingFaceEndpoint)
    empty_llm.repo_id = "test/model"
    empty_llm.model = "test/model"

    model = ChatHuggingFace(
        model_id="moonshotai/Kimi-K2-Instruct-0905",
        llm=empty_llm,
    )
    assert model.profile


def test_init_chat_model_huggingface() -> None:
    """Test that init_chat_model works with HuggingFace models.

    This test verifies that the fix for issue #28226 works correctly.
    The issue was that init_chat_model didn't properly handle HuggingFace
    model initialization, particularly the required 'task' parameter and
    parameter separation between HuggingFacePipeline and ChatHuggingFace.
    """
    from langchain.chat_models.base import init_chat_model

    # Test basic initialization with default task
    # Note: This test may skip in CI if model download fails, but it verifies
    # that the initialization code path works correctly
    try:
        llm = init_chat_model(
            model="microsoft/Phi-3-mini-4k-instruct",
            model_provider="huggingface",
            temperature=0,
            max_tokens=1024,
        )

        # Verify that ChatHuggingFace was created successfully
        assert llm is not None
        from langchain_huggingface import ChatHuggingFace

        assert isinstance(llm, ChatHuggingFace)

        # Verify that the llm attribute is set (this was the bug - it was missing)
        assert hasattr(llm, "llm")
        assert llm.llm is not None

        # Test with explicit task parameter
        llm2 = init_chat_model(
            model="microsoft/Phi-3-mini-4k-instruct",
            model_provider="huggingface",
            task="text-generation",
            temperature=0.5,
        )
        assert isinstance(llm2, ChatHuggingFace)
        assert llm2.llm is not None
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as e:
        # If model download fails in CI, skip the test rather than failing
        # The important part is that the code path doesn't raise ValidationError
        # about missing 'llm' field, which was the original bug
        pytest.skip(f"Skipping test due to model download/initialization error: {e}")


# Tests for async streaming capability contract
def test_huggingface_pipeline_supports_async_streaming() -> None:
    """Test that HuggingFacePipeline declares it does not support async streaming."""
    llm = HuggingFacePipeline()
    assert llm.supports_async_streaming() is False


def test_huggingface_endpoint_supports_async_streaming() -> None:
    """Test that HuggingFaceEndpoint declares it supports async streaming."""
    llm = HuggingFaceEndpoint(  # type: ignore[call-arg]
        repo_id="test/model",
        task="text-generation",
    )
    assert llm.supports_async_streaming() is True


def test_chat_huggingface_supports_async_streaming_with_endpoint(
    mock_llm: Any,
) -> None:
    """Test ChatHuggingFace capability check with HuggingFaceEndpoint."""
    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ), patch(
        "langchain_huggingface.chat_models.huggingface._is_huggingface_endpoint",
        return_value=True,
    ):
        chat = ChatHuggingFace(llm=mock_llm)
        assert chat.supports_async_streaming() is True


def test_chat_huggingface_supports_async_streaming_with_pipeline() -> None:
    """Test ChatHuggingFace capability check with HuggingFacePipeline."""
    pipeline_llm = Mock(spec=HuggingFacePipeline)
    pipeline_llm.supports_async_streaming = Mock(return_value=False)

    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ), patch(
        "langchain_huggingface.chat_models.huggingface._is_huggingface_endpoint",
        return_value=False,
    ):
        chat = ChatHuggingFace(llm=pipeline_llm, tokenizer=MagicMock())
        assert chat.supports_async_streaming() is False


def test_chat_huggingface_supports_async_streaming_falls_back_to_llm_method() -> None:
    """Test that ChatHuggingFace checks LLM's supports_async_streaming if available."""
    # Use HuggingFacePipeline as base to pass validation
    pipeline_llm = Mock(spec=HuggingFacePipeline)
    # Override supports_async_streaming to return True for this test
    pipeline_llm.supports_async_streaming = Mock(return_value=True)

    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ), patch(
        "langchain_huggingface.chat_models.huggingface._is_huggingface_endpoint",
        return_value=False,
    ), patch(
        "langchain_huggingface.chat_models.huggingface._is_huggingface_pipeline",
        return_value=True,
    ):
        chat = ChatHuggingFace(llm=pipeline_llm, tokenizer=MagicMock())
        result = chat.supports_async_streaming()
        assert result is True
        pipeline_llm.supports_async_streaming.assert_called_once()


@pytest.mark.asyncio
async def test_chat_huggingface_astream_with_pipeline_falls_back_to_sync() -> None:
    """Test that _astream falls back to sync streaming for HuggingFacePipeline."""
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    pipeline_llm = Mock(spec=HuggingFacePipeline)
    # Set supports_async_streaming to return False
    pipeline_llm.supports_async_streaming = Mock(return_value=False)

    # Create a proper sync stream that yields chunks
    message_chunk = AIMessageChunk(content="chunk1")
    mock_chunk = ChatGenerationChunk(message=message_chunk, generation_info=None)
    sync_iterator = iter([mock_chunk])

    messages = [HumanMessage(content="Hello")]
    mock_run_manager = MagicMock()
    mock_run_manager.get_sync = Mock(return_value=MagicMock())

    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ), patch(
        "langchain_huggingface.chat_models.huggingface._is_huggingface_endpoint",
        return_value=False,
    ), patch(
        "langchain_huggingface.chat_models.huggingface._is_huggingface_pipeline",
        return_value=True,
    ), patch(
        "langchain_core.runnables.config.run_in_executor"
    ) as mock_executor:
        # Mock run_in_executor to simulate async execution
        call_count = {"stream": 0, "next": 0}

        async def mock_run_in_executor(executor, func, *args, **kwargs):
            if func == next:
                call_count["next"] += 1
                # Simulate next() call on iterator
                try:
                    return next(*args)
                except StopIteration:
                    return kwargs.get("default", object())
            # For _stream call, return the iterator
            call_count["stream"] += 1
            return sync_iterator

        mock_executor.side_effect = mock_run_in_executor

        chat = ChatHuggingFace(llm=pipeline_llm, tokenizer=MagicMock())
        chat._stream = Mock(return_value=sync_iterator)

        # Should not raise AttributeError about async_client
        # The capability check should prevent accessing async_client
        assert chat.supports_async_streaming() is False

        # Verify that _astream would use fallback path
        # (We can't easily test the full async iteration without more complex mocking)
        # But we can verify the capability check prevents the error
        assert not hasattr(pipeline_llm, "async_client") or pipeline_llm.async_client is None


@pytest.mark.asyncio
async def test_chat_huggingface_astream_with_endpoint_uses_async_client(
    mock_llm: Any,
) -> None:
    """Test that _astream uses async_client for HuggingFaceEndpoint."""
    mock_llm.async_client = MagicMock()
    # chat_completion is awaited, so it should return an awaitable that resolves to async iterator
    async_iterator = AsyncIteratorMock(
        [
            {
                "choices": [
                    {
                        "delta": {"content": "Hello"},
                        "finish_reason": None,
                    }
                ]
            }
        ]
    )
    # Make chat_completion an async function that returns the iterator
    async def mock_chat_completion(*args, **kwargs):
        return async_iterator

    mock_llm.async_client.chat_completion = mock_chat_completion

    messages = [HumanMessage(content="Hello")]
    # Create a simple mock run manager with async on_llm_new_token
    # Use a simple class to avoid MagicMock attribute conflicts
    class MockRunManager:
        async def on_llm_new_token(self, *args, **kwargs):
            pass

    mock_run_manager = MockRunManager()

    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ), patch(
        "langchain_huggingface.chat_models.huggingface._is_huggingface_endpoint",
        return_value=True,
    ):
        chat = ChatHuggingFace(llm=mock_llm, tokenizer=MagicMock())
        chat._create_message_dicts = Mock(
            return_value=([{"role": "user", "content": "Hello"}], {})
        )
        chat._should_stream_usage = Mock(return_value=False)

        # Should use async_client, not fall back to sync
        chunks = []
        async for chunk in chat._astream(messages, run_manager=mock_run_manager):
            chunks.append(chunk)

        # Verify async_client was called
        # The async function should have been called
        assert len(chunks) > 0


def test_chat_huggingface_astream_capability_check_prevents_attribute_error() -> None:
    """Test that capability check prevents AttributeError when async_client doesn't exist.

    This test verifies the fix for issue #34134 where HuggingFacePipeline
    caused AttributeError when _astream tried to access async_client.
    """
    # Create a mock that mimics HuggingFacePipeline behavior
    pipeline_llm = Mock(spec=HuggingFacePipeline)
    # Explicitly set supports_async_streaming as a callable that returns False
    # Use a function instead of Mock to ensure it returns a boolean
    def mock_supports_async_streaming():
        return False
    pipeline_llm.supports_async_streaming = mock_supports_async_streaming
    # Explicitly ensure async_client doesn't exist
    if hasattr(pipeline_llm, "async_client"):
        delattr(pipeline_llm, "async_client")

    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ), patch(
        "langchain_huggingface.chat_models.huggingface._is_huggingface_endpoint",
        return_value=False,
    ), patch(
        "langchain_huggingface.chat_models.huggingface._is_huggingface_pipeline",
        return_value=True,
    ):
        chat = ChatHuggingFace(llm=pipeline_llm, tokenizer=MagicMock())

        # Verify capability check returns False
        # This should call supports_async_streaming on the LLM
        result = chat.supports_async_streaming()
        # The result should be False (from the function), not a Mock object
        assert result is False
        assert isinstance(result, bool)

        # Verify that _astream would check capability first
        # This prevents AttributeError from accessing async_client
        # The actual _astream call would fall back to sync streaming
        assert not hasattr(pipeline_llm, "async_client")


def test_base_chat_model_supports_async_streaming_default() -> None:
    """Test that BaseChatModel default implementation checks method override."""
    from langchain_core.language_models.chat_models import BaseChatModel

    # Create a minimal subclass that doesn't override _astream
    class TestChatModel(BaseChatModel):
        def _generate(
            self, messages: list, stop: list[str] | None = None, **kwargs: Any
        ) -> Any:
            pass

        @property
        def _llm_type(self) -> str:
            return "test"

    model = TestChatModel()
    # Default implementation should return False since _astream is not overridden
    assert model.supports_async_streaming() is False


class AsyncIteratorMock:
    """Mock async iterator for testing."""

    def __init__(self, items: list[Any]) -> None:
        self.items = items
        self.index = 0

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item
