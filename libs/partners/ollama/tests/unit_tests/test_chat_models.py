"""Unit tests for ChatOllama."""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from httpx import Client, Request, Response
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import ChatMessage, HumanMessage
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_ollama.chat_models import (
    ChatOllama,
    _parse_arguments_from_tool_call,
    _parse_json_string,
)

MODEL_NAME = "llama3.1"


@contextmanager
def _mock_httpx_client_stream(
    *_args: Any, **_kwargs: Any
) -> Generator[Response, Any, Any]:
    yield Response(
        status_code=200,
        content='{"message": {"role": "assistant", "content": "The meaning ..."}}',
        request=Request(method="POST", url="http://whocares:11434"),
    )


dummy_raw_tool_call = {
    "function": {"name": "test_func", "arguments": ""},
}


class TestChatOllama(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME}


def test__parse_arguments_from_tool_call() -> None:
    """Test that string arguments are preserved as strings in tool call parsing.

    PR #30154
    String-typed tool arguments (like IDs or long strings) were being incorrectly
    processed. The parser should preserve string values as strings rather than
    attempting to parse them as JSON when they're already valid string arguments.

    Use a long string ID to ensure string arguments maintain their original type after
    parsing, which is critical for tools expecting string inputs.
    """
    raw_response = (
        '{"model":"sample-model","message":{"role":"assistant","content":"",'
        '"tool_calls":[{"function":{"name":"get_profile_details",'
        '"arguments":{"arg_1":"12345678901234567890123456"}}}]},"done":false}'
    )
    raw_tool_calls = json.loads(raw_response)["message"]["tool_calls"]
    response = _parse_arguments_from_tool_call(raw_tool_calls[0])
    assert response is not None
    assert isinstance(response["arg_1"], str)
    assert response["arg_1"] == "12345678901234567890123456"


def test__parse_arguments_from_tool_call_with_function_name_metadata() -> None:
    """Test that functionName metadata is filtered out from tool arguments.

    Some models may include metadata like `functionName` in the arguments
    that just echoes the function name. This should be filtered out for
    no-argument tools to return an empty dictionary.
    """
    raw_tool_call_with_metadata = {
        "function": {
            "name": "magic_function_no_args",
            "arguments": {"functionName": "magic_function_no_args"},
        }
    }
    response = _parse_arguments_from_tool_call(raw_tool_call_with_metadata)
    assert response == {}

    # Arguments contain both real args and metadata
    raw_tool_call_mixed = {
        "function": {
            "name": "some_function",
            "arguments": {"functionName": "some_function", "real_arg": "value"},
        }
    }
    response_mixed = _parse_arguments_from_tool_call(raw_tool_call_mixed)
    assert response_mixed == {"real_arg": "value"}

    # functionName has different value (should be preserved)
    raw_tool_call_different = {
        "function": {"name": "function_a", "arguments": {"functionName": "function_b"}}
    }
    response_different = _parse_arguments_from_tool_call(raw_tool_call_different)
    assert response_different == {"functionName": "function_b"}


def test_arbitrary_roles_accepted_in_chatmessages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that `ChatOllama` accepts arbitrary roles in `ChatMessage`."""
    monkeypatch.setattr(Client, "stream", _mock_httpx_client_stream)
    llm = ChatOllama(
        model=MODEL_NAME,
        verbose=True,
        format=None,
    )
    messages = [
        ChatMessage(
            role="somerandomrole",
            content="I'm ok with you adding any role message now!",
        ),
        ChatMessage(role="control", content="thinking"),
        ChatMessage(role="user", content="What is the meaning of life?"),
    ]
    llm.invoke(messages)


@patch("langchain_ollama.chat_models.validate_model")
def test_validate_model_on_init(mock_validate_model: Any) -> None:
    """Test that the model is validated on initialization when requested."""
    ChatOllama(model=MODEL_NAME, validate_model_on_init=True)
    mock_validate_model.assert_called_once()
    mock_validate_model.reset_mock()

    ChatOllama(model=MODEL_NAME, validate_model_on_init=False)
    mock_validate_model.assert_not_called()
    ChatOllama(model=MODEL_NAME)
    mock_validate_model.assert_not_called()


@pytest.mark.parametrize(
    ("input_string", "expected_output"),
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
    malformed_string = "{'key': 'value',,}"  # Double comma is invalid
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
        skip=True,  # We want the original invalid string back
    )
    assert result == malformed_string


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

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_only_response

        llm = ChatOllama(model="test-model")

        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ValueError, match="No data received from Ollama stream"),
        ):
            llm.invoke([HumanMessage("Hello")])

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

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_whitespace_response

        llm = ChatOllama(model="test-model")

        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ValueError, match="No data received from Ollama stream"),
        ):
            llm.invoke([HumanMessage("Hello")])
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

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_then_content_response

        llm = ChatOllama(model="test-model")

        with caplog.at_level(logging.WARNING):
            result = llm.invoke([HumanMessage("Hello")])

        assert "Ollama returned empty response with done_reason='load'" in caplog.text
        assert result.content == "Hello! How can I help you today?"
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

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_with_content_response

        llm = ChatOllama(model="test-model")

        with caplog.at_level(logging.WARNING):
            result = llm.invoke([HumanMessage("Hello")])

        assert result.content == "This is actual content"
        assert result.response_metadata.get("done_reason") == "load"
        assert not caplog.text


def test_none_parameters_excluded_from_options() -> None:
    """Test that None parameters are excluded from the options dict sent to Ollama."""
    response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = response

        # Create ChatOllama with only num_ctx set
        llm = ChatOllama(model="test-model", num_ctx=4096)
        llm.invoke([HumanMessage("Hello")])

        # Verify that chat was called
        assert mock_client.chat.called

        # Get the options dict that was passed to chat
        call_kwargs = mock_client.chat.call_args[1]
        options = call_kwargs.get("options", {})

        # Only num_ctx should be in options, not None parameters
        assert "num_ctx" in options
        assert options["num_ctx"] == 4096

        # These parameters should NOT be in options since they were None
        assert "mirostat" not in options
        assert "mirostat_eta" not in options
        assert "mirostat_tau" not in options
        assert "tfs_z" not in options


def test_all_none_parameters_results_in_empty_options() -> None:
    """Test that when all parameters are None, options dict is empty."""
    response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = response

        # Create ChatOllama with no parameters set
        llm = ChatOllama(model="test-model")
        llm.invoke([HumanMessage("Hello")])

        # Get the options dict that was passed to chat
        call_kwargs = mock_client.chat.call_args[1]
        options = call_kwargs.get("options", {})

        # Options should be empty when no parameters are set
        assert options == {}


def test_explicit_options_dict_preserved() -> None:
    """Test that explicitly provided options dict is preserved and not filtered."""
    response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = response

        llm = ChatOllama(model="test-model")
        # Pass explicit options dict, including None values
        llm.invoke(
            [HumanMessage("Hello")],
            options={"temperature": 0.5, "custom_param": None},
        )

        # Get the options dict that was passed to chat
        call_kwargs = mock_client.chat.call_args[1]
        options = call_kwargs.get("options", {})

        # Explicit options should be preserved as-is
        assert options == {"temperature": 0.5, "custom_param": None}


def test_reasoning_param_passed_to_client() -> None:
    """Test that the reasoning parameter is correctly passed to the Ollama client."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "deepseek-r1",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "message": {"role": "assistant", "content": "I am thinking..."},
                "done": True,
                "done_reason": "stop",
            }
        ]

        # Case 1: reasoning=True in init
        llm = ChatOllama(model="deepseek-r1", reasoning=True)
        llm.invoke([HumanMessage("Hello")])

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["think"] is True

        # Case 2: reasoning=False in init
        llm = ChatOllama(model="deepseek-r1", reasoning=False)
        llm.invoke([HumanMessage("Hello")])

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["think"] is False

        # Case 3: reasoning passed in invoke
        llm = ChatOllama(model="deepseek-r1")
        llm.invoke([HumanMessage("Hello")], reasoning=True)

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["think"] is True


def test_chat_ollama_ignores_strict_arg() -> None:
    """Test that ChatOllama ignores the 'strict' argument."""
    response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = response

        llm = ChatOllama(model="test-model")
        # Invoke with strict=True
        llm.invoke([HumanMessage("Hello")], strict=True)

        # Check that 'strict' was NOT passed to the client
        call_kwargs = mock_client.chat.call_args[1]
        assert "strict" not in call_kwargs


def test_chat_ollama_supports_response_format_json_schema() -> None:
    """Test that ChatOllama correctly maps json_schema response_format to format."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")
        schema = {"type": "object", "properties": {"foo": {"type": "string"}}}
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": schema, "strict": True},
        }

        llm.invoke([HumanMessage("Hello")], response_format=response_format)

        call_kwargs = mock_client.chat.call_args[1]
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("format") == schema


def test_chat_ollama_supports_response_format_json_object() -> None:
    """Test ChatOllama maps json_object response_format to format='json'."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")
        response_format = {"type": "json_object"}

        llm.invoke([HumanMessage("Hello")], response_format=response_format)

        call_kwargs = mock_client.chat.call_args[1]
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("format") == "json"
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")
        response_format = {"type": "json_object"}

        # User passes BOTH format param and response_format
        llm.invoke(
            [HumanMessage("Hello")],
            format="some_custom_format",
            response_format=response_format,
        )

        call_kwargs = mock_client.chat.call_args[1]
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("format") == "some_custom_format"


def test_chat_ollama_ignores_invalid_response_format_type() -> None:
    """Test that ChatOllama ignores non-dict response_format without crashing."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")
        # Pass a list (invalid type) instead of a dict
        response_format = ["invalid_type"]

        # This should not raise AttributeError
        llm.invoke([HumanMessage("Hello")], response_format=response_format)

        call_kwargs = mock_client.chat.call_args[1]
        # response_format should be removed even if invalid type
        assert "response_format" not in call_kwargs
        # Format should remain None since we couldn't parse it
        assert call_kwargs.get("format") is None
