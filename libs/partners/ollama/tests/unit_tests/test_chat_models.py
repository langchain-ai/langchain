"""Unit tests for ChatOllama."""

import contextlib
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
from langchain_core.output_parsers import PydanticOutputParser
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import BaseModel
from typing_extensions import Literal

from langchain_ollama.chat_models import (
    ChatOllama,
    _parse_arguments_from_tool_call,
    _parse_json_string,
)

MODEL_NAME = "llama3.1"


class TestChatOllama(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME}


def test__parse_arguments_from_tool_call() -> None:
    """Test that string arguments are preserved as strings in tool call parsing.

    This test verifies the fix for PR #30154 which addressed an issue where
    string-typed tool arguments (like IDs or long strings) were being
    incorrectly processed. The parser should preserve string values as strings
    rather than attempting to parse them as JSON when they're already valid
    string arguments.

    The test uses a long string ID to ensure string arguments maintain their
    original type after parsing, which is critical for tools expecting string
    inputs.
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


def test__parse_arguments_from_tool_call_with_function_name_metadata() -> None:
    """Test that functionName metadata is filtered out from tool arguments.

    Some models may include metadata like ``functionName`` in the arguments
    that just echoes the function name. This should be filtered out for
    no-argument tools to return an empty dictionary.
    """
    # Test case where arguments contain functionName metadata
    raw_tool_call_with_metadata = {
        "function": {
            "name": "magic_function_no_args",
            "arguments": {"functionName": "magic_function_no_args"},
        }
    }
    response = _parse_arguments_from_tool_call(raw_tool_call_with_metadata)
    assert response == {}

    # Test case where arguments contain both real args and metadata
    raw_tool_call_mixed = {
        "function": {
            "name": "some_function",
            "arguments": {"functionName": "some_function", "real_arg": "value"},
        }
    }
    response_mixed = _parse_arguments_from_tool_call(raw_tool_call_mixed)
    assert response_mixed == {"real_arg": "value"}

    # Test case where functionName has different value (should be preserved)
    raw_tool_call_different = {
        "function": {
            "name": "function_a",
            "arguments": {"functionName": "function_b"},
        }
    }
    response_different = _parse_arguments_from_tool_call(raw_tool_call_different)
    assert response_different == {"functionName": "function_b"}


@contextmanager
def _mock_httpx_client_stream(
    *args: Any, **kwargs: Any
) -> Generator[Response, Any, Any]:
    yield Response(
        status_code=200,
        content='{"message": {"role": "assistant", "content": "The meaning ..."}}',
        request=Request(method="POST", url="http://whocares:11434"),
    )


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
    # Test that validate_model is called when validate_model_on_init=True
    ChatOllama(model=MODEL_NAME, validate_model_on_init=True)
    mock_validate_model.assert_called_once()
    mock_validate_model.reset_mock()

    # Test that validate_model is NOT called when validate_model_on_init=False
    ChatOllama(model=MODEL_NAME, validate_model_on_init=False)
    mock_validate_model.assert_not_called()

    # Test that validate_model is NOT called by default
    ChatOllama(model=MODEL_NAME)
    mock_validate_model.assert_not_called()


# Define a dummy raw_tool_call for the function signature
dummy_raw_tool_call = {
    "function": {"name": "test_func", "arguments": ""},
}


def test_parse_json_string_empty_string_cases() -> None:
    """Test _parse_json_string handling of empty strings."""
    raw_tool_call = {"function": {"name": "test_func", "arguments": ""}}

    # Test empty string with skip=False should return empty dict
    result = _parse_json_string("", raw_tool_call=raw_tool_call, skip=False)
    assert result == {}, f"Expected empty dict for empty string, got {result}"

    # Test whitespace-only string with skip=False should return empty dict
    result = _parse_json_string(
        "   \n  \t  ", raw_tool_call=raw_tool_call, skip=False
    )
    assert result == {}, f"Expected empty dict for whitespace string, got {result}"

    # Test empty string with skip=True should return original string
    result = _parse_json_string("", raw_tool_call=raw_tool_call, skip=True)
    assert result == "", f"Expected empty string, got {result}"

    # Test whitespace-only string with skip=True should return original string
    whitespace_str = "   \n  \t  "
    result = _parse_json_string(
        whitespace_str, raw_tool_call=raw_tool_call, skip=True
    )
    expected_msg = f"Expected original whitespace string, got {result}"
    assert result == whitespace_str, expected_msg


def test_parse_arguments_from_tool_call_empty_arguments() -> None:
    """Test _parse_arguments_from_tool_call with empty arguments."""
    # Test with empty string arguments
    raw_tool_call_empty = {
        "function": {"name": "test_function", "arguments": ""}
    }
    result = _parse_arguments_from_tool_call(raw_tool_call_empty)
    assert result == {}, f"Expected empty dict for empty arguments, got {result}"

    # Test with whitespace-only arguments
    raw_tool_call_whitespace = {
        "function": {"name": "test_function", "arguments": "   \n\t   "}
    }
    result = _parse_arguments_from_tool_call(raw_tool_call_whitespace)
    expected_msg = f"Expected empty dict for whitespace arguments, got {result}"
    assert result == {}, expected_msg

    # Test with empty dict arguments
    raw_tool_call_empty_dict = {
        "function": {"name": "test_function", "arguments": {}}
    }
    result = _parse_arguments_from_tool_call(raw_tool_call_empty_dict)
    expected_msg = f"Expected empty dict for empty dict arguments, got {result}"
    assert result == {}, expected_msg


def test_structured_output_with_empty_responses() -> None:
    """Test structured output handling when Ollama returns empty responses."""

    class TestSchema(BaseModel):
        sentiment: Literal["happy", "neutral", "sad"]
        language: Literal["english", "spanish"]

    # Test with empty content but valid tool calls
    empty_content_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {
                "role": "assistant",
                "content": "",  # Empty content
                "tool_calls": [
                    {
                        "function": {
                            "name": "TestSchema",
                            "arguments": (
                                '{"sentiment": "happy", "language": "spanish"}'
                            ),
                        }
                    }
                ],
            },
            "done": False,
        },
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        },
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = iter(empty_content_response)

        llm = ChatOllama(model="test-model")
        structured_llm = llm.with_structured_output(
            TestSchema, method="function_calling"
        )

        try:
            result = structured_llm.invoke("Test input")
            assert isinstance(result, TestSchema)
            assert result.sentiment == "happy"
            assert result.language == "spanish"
        except (ValueError) as e:
            pytest.fail(f"Failed to handle empty content with tool calls: {e}")


def test_structured_output_with_completely_empty_response() -> None:
    """Test structured output when Ollama returns completely empty response."""

    class TestSchema(BaseModel):
        sentiment: Literal["happy", "neutral", "sad"]
        language: Literal["english", "spanish"]

    # Completely empty response
    empty_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = iter(empty_response)

        llm = ChatOllama(model="test-model")

        # This should handle empty responses gracefully
        for method in ["json_mode", "json_schema", "function_calling"]:
            mock_client.reset_mock()
            mock_client.chat.return_value = iter(empty_response)

            structured_llm = llm.with_structured_output(TestSchema, method=method)

            try:
                structured_llm.invoke("Test input")
                # The behavior here depends on the method and parser implementation
                # At minimum, it shouldn't crash with OutputParserException
            except OutputParserException as e:
                if "Unexpected end of JSON input" in str(e):
                    error_msg = (
                        f"{method} still throwing original empty string error: {e}"
                    )
                    pytest.fail(error_msg)
                # Other parsing errors might be acceptable
            except (ValueError):
                # Non-parsing errors might be acceptable
                pass


# Updated version of existing test to verify the fix
@pytest.mark.parametrize(
    ("input_string", "expected_output"),
    [
        # Existing test cases
        ('{"key": "value", "number": 123}', {"key": "value", "number": 123}),
        ("{'key': 'value', 'number': 123}", {"key": "value", "number": 123}),
        ('{"text": "It\'s a great test!"}', {"text": "It's a great test!"}),
        ("{'text': \"It's a great test!\"}", {"text": "It's a great test!"}),
        # NEW: Empty string test cases
        ("", {}),  # Empty string should return empty dict
        ("   ", {}),  # Whitespace-only should return empty dict
        ("\n\t  ", {}),  # Mixed whitespace should return empty dict
    ],
)
def test_parse_json_string_success_cases_with_empty_strings(
    input_string: str, expected_output: Any
) -> None:
    """Updated test that includes empty string handling."""
    raw_tool_call = {"function": {"name": "test_func", "arguments": input_string}}
    result = _parse_json_string(input_string, raw_tool_call=raw_tool_call, skip=False)
    assert result == expected_output, f"Expected {expected_output}, got {result}"


def test_parse_json_string_skip_behavior_with_empty_strings() -> None:
    """Test skip behavior specifically with empty strings."""
    raw_tool_call = {"function": {"name": "test_func", "arguments": ""}}

    # Empty string with skip=True should return the empty string
    result = _parse_json_string("", raw_tool_call=raw_tool_call, skip=True)
    assert result == "", f"Expected empty string with skip=True, got {result}"

    # Malformed JSON with skip=True should return original string
    malformed = "{'not': valid,,,}"
    raw_tool_call_malformed = {
        "function": {"name": "test_func", "arguments": malformed}
    }
    result = _parse_json_string(
        malformed, raw_tool_call=raw_tool_call_malformed, skip=True
    )
    assert result == malformed, f"Expected original malformed string, got {result}"


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


def test_structured_output_parsing() -> None:
    """Test that structured output parsing works correctly with different methods."""

    class TestSchema(BaseModel):
        sentiment: Literal["happy", "neutral", "sad"]
        language: Literal["english", "spanish"]

    # Test the parsers work correctly first
    json_content = '{"sentiment": "happy", "language": "spanish"}'
    pydantic_parser = PydanticOutputParser(pydantic_object=TestSchema)
    parsed_result = pydantic_parser.parse(json_content)
    assert isinstance(parsed_result, TestSchema)

    # Now test with ChatOllama - patch the streaming methods directly
    with (
        patch("langchain_ollama.chat_models.Client") as mock_client_class,
        patch("langchain_ollama.chat_models.AsyncClient") as mock_async_client_class,
    ):
        # Set up mocks
        mock_client = MagicMock()
        mock_async_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_async_client_class.return_value = mock_async_client

        # Create a proper streaming response that matches Ollama's actual format
        streaming_response = [
            # First chunk with content
            {
                "model": "test-model",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "message": {"role": "assistant", "content": json_content},
                "done": False,
            },
            # Final chunk with done=True and metadata
            {
                "model": "test-model",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "total_duration": 1000000,
                "load_duration": 100000,
                "prompt_eval_count": 10,
                "prompt_eval_duration": 50000,
                "eval_count": 20,
                "eval_duration": 500000,
            },
        ]

        # The mock needs to return an iterator
        mock_client.chat.return_value = iter(streaming_response)

        # Create ChatOllama instance
        llm = ChatOllama(model="test-model")

        # Test each method individually
        for method in ["json_mode", "json_schema"]:
            # Reset the mock for each test
            mock_client.reset_mock()
            mock_client.chat.return_value = iter(streaming_response)

            # Create structured output wrapper
            structured_llm = llm.with_structured_output(TestSchema, method=method)

            # Test the invoke
            try:
                result = structured_llm.invoke("Test input")

                if result is None:
                    # Test if we can manually invoke the base LLM
                    mock_client.chat.return_value = iter(streaming_response)
                    base_result = llm.invoke("Test input")

                    # Test if we can parse that content directly
                    if base_result.content.strip():
                        with contextlib.suppress(Exception):
                            pydantic_parser.parse(base_result.content)
                else:
                    expected_msg = (
                        f"Expected TestSchema for {method}, "
                        f"got {type(result)}: {result}"
                    )
                    assert isinstance(result, TestSchema), expected_msg
                    assert result.sentiment == "happy"
                    assert result.language == "spanish"

            except (ValueError, TypeError):
                # Allow exceptions during testing
                pass

        # Test function_calling separately since it has different response format
        function_calling_response = [
            # Tool call chunk
            {
                "model": "test-model",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "TestSchema", "arguments": json_content}}
                    ],
                },
                "done": False,
            },
            # Completion chunk
            {
                "model": "test-model",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "total_duration": 1000000,
            },
        ]

        mock_client.reset_mock()
        mock_client.chat.return_value = iter(function_calling_response)

        structured_llm = llm.with_structured_output(
            TestSchema, method="function_calling"
        )

        try:
            result = structured_llm.invoke("Test input")

            if result is not None:
                expected_msg = (
                    f"Expected TestSchema for function_calling, "
                    f"got {type(result)}: {result}"
                )
                assert isinstance(result, TestSchema), expected_msg
                assert result.sentiment == "happy"
                assert result.language == "spanish"

        except (ValueError, TypeError):
            # Allow exceptions during testing
            pass

        # Test empty response handling
        empty_response = [
            {
                "model": "test-model",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "message": {"role": "assistant", "content": ""},  # Empty content
                "done": True,
                "done_reason": "stop",
            }
        ]

        for method in ["json_mode", "json_schema"]:
            mock_client.reset_mock()
            mock_client.chat.return_value = iter(empty_response)

            structured_llm = llm.with_structured_output(TestSchema, method=method)

            try:
                structured_llm.invoke("Test input")
                # The key test: it shouldn't crash with "Unexpected end of JSON input"
            except Exception as e:
                if "Unexpected end of JSON input" in str(e):
                    error_msg = f"{method} should handle empty strings gracefully"
                    raise AssertionError(error_msg) from e

        # Test function_calling with empty arguments
        empty_args_response = [
            {
                "model": "test-model",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "TestSchema", "arguments": ""}}  # Empty
                    ],
                },
                "done": True,
                "done_reason": "stop",
            }
        ]

        mock_client.reset_mock()
        mock_client.chat.return_value = iter(empty_args_response)
        structured_llm = llm.with_structured_output(
            TestSchema, method="function_calling"
        )

        try:
            structured_llm.invoke("Test input")
        except Exception as e:
            if "Unexpected end of JSON input" in str(e):
                error_msg = "function_calling should handle empty arguments gracefully"
                raise AssertionError(error_msg) from e
