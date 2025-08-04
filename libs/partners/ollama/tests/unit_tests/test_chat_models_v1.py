"""Unit tests for ChatOllamaV1."""

import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages.content_blocks import (
    create_image_block,
    create_text_block,
)
from langchain_core.messages.v1 import AIMessage as AIMessageV1
from langchain_core.messages.v1 import HumanMessage as HumanMessageV1
from langchain_core.messages.v1 import MessageV1
from langchain_core.messages.v1 import SystemMessage as SystemMessageV1
from langchain_tests.unit_tests.chat_models_v1 import ChatModelV1UnitTests

from langchain_ollama._compat import (
    _convert_chunk_to_v1,
    _convert_from_v1_to_ollama_format,
    _convert_to_v1_from_ollama_format,
)
from langchain_ollama.chat_models_v1 import (
    ChatOllamaV1,
    _parse_arguments_from_tool_call,
    _parse_json_string,
)

MODEL_NAME = "llama3.1"


class TestMessageConversion:
    """Test v1 message conversion utilities."""

    def test_convert_human_message_v1_text_only(self) -> None:
        """Test converting HumanMessageV1 with text content."""
        message = HumanMessageV1("Hello world")

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "user"
        assert result["content"] == "Hello world"
        assert result["images"] == []

    def test_convert_ai_message_v1(self) -> None:
        """Test converting AIMessageV1 with text content."""
        message = AIMessageV1("Hello! How can I help?")

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "assistant"
        assert result["content"] == "Hello! How can I help?"

    def test_convert_system_message_v1(self) -> None:
        """Test converting SystemMessageV1."""
        message = SystemMessageV1("You are a helpful assistant.")

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "system"
        assert result["content"] == "You are a helpful assistant."

    def test_convert_human_message_v1_with_image(self) -> None:
        """Test converting HumanMessageV1 with text and image content.

        Each uses `_convert_from_v1_to_ollama_format` to ensure
        that the conversion handles both text and image blocks correctly. Thus, we don't
        need additional tests for other message types that also use this function.

        """
        message_a = HumanMessageV1(
            content=[
                create_text_block("Describe this image:"),
                create_image_block(base64="base64imagedata"),
            ]
        )

        result_a = _convert_from_v1_to_ollama_format(message_a)

        assert result_a["role"] == "user"
        assert result_a["content"] == "Describe this image:"
        assert result_a["images"] == ["base64imagedata"]

        # Make sure multiple images are handled correctly
        message_b = HumanMessageV1(
            content=[
                create_text_block("Describe this image:"),
                create_image_block(base64="base64imagedata"),
                create_image_block(base64="base64dataimage"),
            ]
        )

        result_b = _convert_from_v1_to_ollama_format(message_b)

        assert result_b["role"] == "user"
        assert result_b["content"] == "Describe this image:"
        assert result_b["images"] == ["base64imagedata", "base64dataimage"]

    def test_convert_from_ollama_format(self) -> None:
        """Test converting Ollama response to `AIMessageV1`."""
        ollama_response = {
            "model": MODEL_NAME,
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
            },
            "done": True,
            "done_reason": "stop",
            "total_duration": 1000000,
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

        result = _convert_to_v1_from_ollama_format(ollama_response)

        assert isinstance(result, AIMessageV1)
        assert len(result.content) == 1
        assert result.content[0].get("type") == "text"
        assert result.content[0].get("text") == "Hello! How can I help you today?"
        assert result.response_metadata.get("model_name") == MODEL_NAME
        assert result.response_metadata.get("done") is True

    def test_convert_chunk_to_v1(self) -> None:
        """Test converting Ollama streaming chunk to `AIMessageChunkV1`."""
        chunk = {
            "model": MODEL_NAME,
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Hello"},
            "done": False,
        }

        result = _convert_chunk_to_v1(chunk)

        assert len(result.content) == 1
        assert result.content[0].get("type") == "text"
        assert result.content[0].get("text") == "Hello"

    def test_convert_empty_content(self) -> None:
        """Test converting empty content blocks."""
        message = HumanMessageV1(content=[])

        result = _convert_from_v1_to_ollama_format(message)

        assert result["role"] == "user"
        assert result["content"] == ""
        assert result["images"] == []


class TestChatOllamaV1(ChatModelV1UnitTests):
    """Test `ChatOllamaV1`."""

    @property
    def chat_model_class(self) -> type[ChatOllamaV1]:
        return ChatOllamaV1

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME}

    @property
    def has_tool_calling(self) -> bool:
        """`ChatOllamaV1` supports tool calling (e.g., `qwen3` models)."""
        return True

    @property
    def has_tool_choice(self) -> bool:
        """`ChatOllamaV1` supports tool choice parameter."""
        return True

    @property
    def has_structured_output(self) -> bool:
        """`ChatOllamaV1` supports structured output via `with_structured_output`."""
        return True

    @property
    def supports_image_content_blocks(self) -> bool:
        """`ChatOllamaV1` supports image content blocks (e.g., `gemma3`)."""
        return True

    @property
    def supports_reasoning_content_blocks(self) -> bool:
        """`ChatOllamaV1` supports reasoning/thinking content blocks (e.g., `qwen3`)."""
        return True

    @property
    def returns_usage_metadata(self) -> bool:
        """`ChatOllamaV1` returns usage metadata with token counts."""
        return True

    @property
    def supports_non_standard_blocks(self) -> bool:
        """Override to indicate Ollama doesn't support non-standard content blocks."""
        return False

    @pytest.fixture
    def model(self) -> Iterator[ChatOllamaV1]:
        """Create a ChatOllamaV1 instance for testing."""
        sync_patcher = patch("langchain_ollama.chat_models_v1.Client")
        async_patcher = patch("langchain_ollama.chat_models_v1.AsyncClient")

        mock_sync_client_class = sync_patcher.start()
        mock_async_client_class = async_patcher.start()

        mock_sync_client = MagicMock()
        mock_async_client = MagicMock()

        mock_sync_client_class.return_value = mock_sync_client
        mock_async_client_class.return_value = mock_async_client

        def mock_chat_response(*_args: Any, **_kwargs: Any) -> Iterator[dict[str, Any]]:
            return iter(
                [
                    {
                        "model": MODEL_NAME,
                        "created_at": "2024-01-01T00:00:00Z",
                        "message": {"role": "assistant", "content": "Test response"},
                        "done": True,
                        "done_reason": "stop",
                    }
                ]
            )

        async def mock_async_chat_iterator(
            *_args: Any, **_kwargs: Any
        ) -> AsyncIterator[dict[str, Any]]:
            for item in mock_chat_response(*_args, **_kwargs):
                yield item

        mock_sync_client.chat.side_effect = mock_chat_response
        mock_async_client.chat.side_effect = mock_async_chat_iterator

        model_instance = self.chat_model_class(**self.chat_model_params)
        yield model_instance
        sync_patcher.stop()
        async_patcher.stop()

    def test_initialization(self) -> None:
        """Test `ChatOllamaV1` initialization."""
        with (
            patch("langchain_ollama.chat_models_v1.Client"),
            patch("langchain_ollama.chat_models_v1.AsyncClient"),
        ):
            llm = ChatOllamaV1(model=MODEL_NAME)

            assert llm.model == MODEL_NAME
            assert llm._llm_type == "chat-ollama-v1"

    def test_chat_params(self) -> None:
        """Test `_chat_params()`."""
        with (
            patch("langchain_ollama.chat_models_v1.Client"),
            patch("langchain_ollama.chat_models_v1.AsyncClient"),
        ):
            llm = ChatOllamaV1(model=MODEL_NAME, temperature=0.7)

            messages: list[MessageV1] = [HumanMessageV1("Hello")]

            params = llm._chat_params(messages)

            assert params["model"] == MODEL_NAME
            assert len(params["messages"]) == 1
            assert params["messages"][0]["role"] == "user"
            assert params["messages"][0]["content"] == "Hello"

            # Ensure options carry over
            assert params["options"].temperature == 0.7

    def test_ls_params(self) -> None:
        """Test LangSmith parameters."""
        with (
            patch("langchain_ollama.chat_models_v1.Client"),
            patch("langchain_ollama.chat_models_v1.AsyncClient"),
        ):
            llm = ChatOllamaV1(model=MODEL_NAME, temperature=0.5)

            ls_params = llm._get_ls_params()

            assert ls_params.get("ls_provider") == "ollama"
            assert ls_params.get("ls_model_name") == MODEL_NAME
            assert ls_params.get("ls_model_type") == "chat"
            assert ls_params.get("ls_temperature") == 0.5

    def test_bind_tools_basic(self) -> None:
        """Test basic tool binding functionality."""
        with (
            patch("langchain_ollama.chat_models_v1.Client"),
            patch("langchain_ollama.chat_models_v1.AsyncClient"),
        ):
            llm = ChatOllamaV1(model=MODEL_NAME)

            def test_tool(query: str) -> str:
                """A test tool."""
                return f"Result for: {query}"

            bound_llm = llm.bind_tools([test_tool])

            # Should return a bound model
            assert bound_llm is not None


# Missing: `test_arbitrary_roles_accepted_in_chatmessages`
# Not brought over since it would appear that it's just a workaround to `think=True`
# But can be added if needed in the future.


@patch("langchain_ollama.chat_models_v1.validate_model")
@patch("langchain_ollama.chat_models_v1.Client")
def test_validate_model_on_init(
    mock_client_class: Any, mock_validate_model: Any
) -> None:
    """Test that the model is validated on initialization when requested."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Test that validate_model is called when validate_model_on_init=True
    ChatOllamaV1(model=MODEL_NAME, validate_model_on_init=True)
    mock_validate_model.assert_called_once()
    mock_validate_model.reset_mock()

    # Test that validate_model is NOT called when validate_model_on_init=False
    ChatOllamaV1(model=MODEL_NAME, validate_model_on_init=False)
    mock_validate_model.assert_not_called()

    # Test that validate_model is NOT called by default
    ChatOllamaV1(model=MODEL_NAME)
    mock_validate_model.assert_not_called()


# Define a dummy raw_tool_call for the function signature
dummy_raw_tool_call = {
    "function": {"name": "test_func", "arguments": ""},
}


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        # Case 1: Standard double-quoted JSON
        ('{"key": "value", "number": 123}', {"key": "value", "number": 123}),
        # Case 2: Single-quoted string (the original bug)
        ("{'key': 'value', 'number': 123}", {"key": "value", "number": 123}),
        # Case 3: String with an internal apostrophe
        ('{"text": "It\'s a great test!"}', {"text": "It's a great test!"}),
        # Case 4: Mixed quotes that ast can handle
        ("{'text': \"It's a great test!\"}", {"text": "It's a great test!"}),
    ],
)
def test_parse_json_string_success_cases(
    input_string: str, expected_output: Any
) -> None:
    """Tests that `_parse_json_string` correctly parses valid and fixable strings."""
    raw_tool_call = {"function": {"name": "test_func", "arguments": input_string}}
    result = _parse_json_string(input_string, raw_tool_call=raw_tool_call, skip=False)
    assert result == expected_output


def test_parse_json_string_failure_case_raises_exception() -> None:
    """Tests that `_parse_json_string` raises an exception for malformed strings."""
    malformed_string = "{'key': 'value',,}"
    raw_tool_call = {"function": {"name": "test_func", "arguments": malformed_string}}
    with pytest.raises(OutputParserException):
        _parse_json_string(
            malformed_string,
            raw_tool_call=raw_tool_call,
            skip=False,
        )


def test_parse_json_string_skip_returns_input_on_failure() -> None:
    """Tests that `skip=True` returns the original string on parse failure."""
    malformed_string = "{'not': valid,,,}"
    raw_tool_call = {"function": {"name": "test_func", "arguments": malformed_string}}
    result = _parse_json_string(
        malformed_string,
        raw_tool_call=raw_tool_call,
        skip=True,
    )
    assert result == malformed_string


def test__parse_arguments_from_tool_call() -> None:
    """Test that string arguments are preserved as strings in tool call parsing.

    This test verifies the fix for PR #30154 which addressed an issue where
    string-typed tool arguments (like IDs or long strings) were being incorrectly
    processed. The parser should preserve string values as strings rather than
    attempting to parse them as JSON when they're already valid string arguments.

    The test uses a long string ID to ensure string arguments maintain their
    original type after parsing, which is critical for tools expecting string inputs.
    """
    raw_response = '{"model":"sample-model","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_profile_details","arguments":{"arg_1":"12345678901234567890123456"}}}]},"done":false}'  # noqa: E501
    raw_tool_calls = json.loads(raw_response)["message"]["tool_calls"]
    response = _parse_arguments_from_tool_call(raw_tool_calls[0])
    assert response is not None
    assert isinstance(response["arg_1"], str)


def test_load_response_with_empty_content_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that load responses with empty content log a warning and are skipped."""
    load_only_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": ""},
        }
    ]

    with patch("langchain_ollama.chat_models_v1.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = iter(load_only_response)

        llm = ChatOllamaV1(model="test-model")

        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ValueError, match="No generations found in stream"),
        ):
            llm.invoke([HumanMessageV1("Hello")])

        assert "Ollama returned empty response with done_reason='load'" in caplog.text


def test_load_response_with_whitespace_content_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test load responses w/ only whitespace content log a warning and are skipped."""
    load_whitespace_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": "   \n  \t  "},
        }
    ]

    with patch("langchain_ollama.chat_models_v1.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = iter(load_whitespace_response)

        llm = ChatOllamaV1(model="test-model")

        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ValueError, match="No generations found in stream"),
        ):
            llm.invoke([HumanMessageV1("Hello")])
        assert "Ollama returned empty response with done_reason='load'" in caplog.text


def test_load_followed_by_content_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test load responses log a warning and are skipped when followed by content."""
    load_then_content_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": ""},
        },
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:01.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
            },
        },
    ]

    with patch("langchain_ollama.chat_models_v1.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = iter(load_then_content_response)

        llm = ChatOllamaV1(model="test-model")

        with caplog.at_level(logging.WARNING):
            result = llm.invoke([HumanMessageV1("Hello")])

        assert "Ollama returned empty response with done_reason='load'" in caplog.text
        assert len(result.content) == 1
        assert result.text == "Hello! How can I help you today?"
        assert result.response_metadata.get("done_reason") == "stop"


def test_load_response_with_actual_content_is_not_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test load responses with actual content are NOT skipped and log no warning."""
    load_with_content_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": "This is actual content"},
        }
    ]

    with patch("langchain_ollama.chat_models_v1.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = iter(load_with_content_response)

        llm = ChatOllamaV1(model="test-model")

        with caplog.at_level(logging.WARNING):
            result = llm.invoke([HumanMessageV1("Hello")])

        assert len(result.content) == 1
        assert result.text == "This is actual content"
        assert result.response_metadata.get("done_reason") == "load"
        assert not caplog.text
