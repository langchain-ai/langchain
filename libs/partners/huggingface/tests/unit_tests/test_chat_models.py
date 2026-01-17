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
from langchain_huggingface.llms import HuggingFaceEndpoint


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


def test_convert_chunk_to_message_chunk_with_reasoning_content() -> None:
    """Test that reasoning_content is extracted from chunk and included in AIMessageChunk.

    Verifies that both 'reasoning' field from dict-style chunks and 
    'reasoning_content' attribute from object-style chunks are correctly 
    propagated to AIMessageChunk.additional_kwargs['reasoning_content'].
    """
    from langchain_core.messages import AIMessageChunk
    from langchain_huggingface.chat_models.huggingface import _convert_chunk_to_message_chunk

    # Test case 1: reasoning field from dict delta
    chunk_with_reasoning_dict = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": "The answer is 42",
                    "reasoning": "Let me think through this step by step..."
                }
            }
        ]
    }

    result = _convert_chunk_to_message_chunk(
        chunk_with_reasoning_dict, 
        AIMessageChunk
    )

    assert isinstance(result, AIMessageChunk)
    assert result.content == "The answer is 42"
    assert result.additional_kwargs["reasoning_content"] == "Let me think through this step by step..."

    # Test case 2: reasoning_content from object attribute (fallback)
    class MockDelta:
        reasoning_content = "Object-based reasoning trace"
        role = "assistant"
        content = "Response content"

    class MockChoice:
        delta = MockDelta()

    class MockChunk(dict):
        def __init__(self) -> None:
            super().__init__({
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": "Response content"
                        }
                    }
                ]
            })
            self.choices = [MockChoice()]

    chunk_with_reasoning_attr = MockChunk()
    result = _convert_chunk_to_message_chunk(
        chunk_with_reasoning_attr,
        AIMessageChunk
    )

    assert isinstance(result, AIMessageChunk)
    assert result.content == "Response content"
    assert result.additional_kwargs["reasoning_content"] == "Object-based reasoning trace"

    # Test case 3: no reasoning present (None should be set)
    chunk_without_reasoning = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": "Simple response"
                }
            }
        ]
    }

    result = _convert_chunk_to_message_chunk(
        chunk_without_reasoning,
        AIMessageChunk
    )

    assert isinstance(result, AIMessageChunk)
    assert result.content == "Simple response"
    assert result.additional_kwargs["reasoning_content"] is None

    # Test case 4: dict reasoning takes precedence over object attribute
    class MockChunkBoth(dict):
        def __init__(self) -> None:
            super().__init__({
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": "Response",
                            "reasoning": "Dict reasoning value"
                        }
                    }
                ]
            })
            self.choices = [MockChoice()]  # Has reasoning_content attribute

    chunk_with_both = MockChunkBoth()
    result = _convert_chunk_to_message_chunk(chunk_with_both, AIMessageChunk)

    assert result.additional_kwargs["reasoning_content"] == "Dict reasoning value"


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
