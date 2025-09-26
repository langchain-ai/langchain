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
    LenientJsonExtractor,
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
        "function": {"name": "function_a", "arguments": {"functionName": "function_b"}}
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


class TestLenientJsonExtractor:
    """Test the LenientJsonExtractor class for handling reasoning-prefixed outputs."""

    def test_parse_clean_json(self) -> None:
        """Test parsing clean JSON without any prefixes."""
        parser = LenientJsonExtractor()
        result = parser.parse('{"answer": "They weigh the same", "justification": "Both weigh one pound"}')
        assert result == {"answer": "They weigh the same", "justification": "Both weigh one pound"}

    def test_parse_json_with_reasoning_prefix(self) -> None:
        """Test parsing JSON that has <think>...</think> reasoning prefix."""
        parser = LenientJsonExtractor()
        text = '''<think>Let me think about this. A pound of bricks vs a pound of feathers. Both are one pound, so they should weigh the same.</think>
{"answer": "They weigh the same", "justification": "Both weigh one pound"}'''
        result = parser.parse(text)
        assert result == {"answer": "They weigh the same", "justification": "Both weigh one pound"}

    def test_parse_json_with_fenced_block(self) -> None:
        """Test parsing JSON from a ```json fenced code block."""
        parser = LenientJsonExtractor()
        text = '''Here's the answer:
```json
{"answer": "They weigh the same", "justification": "Both weigh one pound"}
```
This should be the correct response.'''
        result = parser.parse(text)
        assert result == {"answer": "They weigh the same", "justification": "Both weigh one pound"}

    def test_parse_json_with_reasoning_and_fenced_block(self) -> None:
        """Test parsing JSON from fenced block with reasoning prefix."""
        parser = LenientJsonExtractor()
        text = '''<think>This is a classic riddle. Let me work through it step by step.</think>
Here's my analysis:
```json
{"answer": "They weigh the same", "justification": "Both weigh one pound"}
```
Hope this helps!'''
        result = parser.parse(text)
        assert result == {"answer": "They weigh the same", "justification": "Both weigh one pound"}

    def test_parse_json_with_case_insensitive_reasoning(self) -> None:
        """Test parsing JSON with case-insensitive reasoning tags."""
        parser = LenientJsonExtractor()
        text = '''<THINK>Let me think about this carefully.</THINK>
{"answer": "They weigh the same", "justification": "Both weigh one pound"}'''
        result = parser.parse(text)
        assert result == {"answer": "They weigh the same", "justification": "Both weigh one pound"}

    def test_parse_json_multiline_reasoning(self) -> None:
        """Test parsing JSON with multiline reasoning content."""
        parser = LenientJsonExtractor()
        text = '''<think>
This is a classic physics question.
Let me think step by step:
1. A pound is a unit of weight
2. Both items are one pound
3. Therefore they weigh the same
</think>
{"answer": "They weigh the same", "justification": "Both weigh one pound"}'''
        result = parser.parse(text)
        assert result == {"answer": "They weigh the same", "justification": "Both weigh one pound"}

    def test_parse_no_json_raises_exception(self) -> None:
        """Test that parsing text without JSON raises OutputParserException."""
        parser = LenientJsonExtractor()
        text = "This is just plain text with no JSON content at all."

        with pytest.raises(OutputParserException) as exc_info:
            parser.parse(text)

        assert "No JSON object found in model output." in str(exc_info.value)
        assert exc_info.value.llm_output == text

    def test_parse_invalid_json_raises_exception(self) -> None:
        """Test that parsing text with invalid JSON raises OutputParserException."""
        parser = LenientJsonExtractor()
        text = '{"invalid": json, "missing": quotes}'

        with pytest.raises(OutputParserException) as exc_info:
            parser.parse(text)

        assert "No JSON object found in model output." in str(exc_info.value)
        assert exc_info.value.llm_output == text

    def test_parse_json_with_nested_objects(self) -> None:
        """Test parsing complex JSON with nested objects."""
        parser = LenientJsonExtractor()
        text = '''<think>This requires careful analysis.</think>
{"question": {"type": "riddle", "category": "physics"}, "answer": {"value": "same", "unit": "pound"}, "explanation": {"reasoning": "weight is identical"}}'''
        result = parser.parse(text)
        expected = {
            "question": {"type": "riddle", "category": "physics"},
            "answer": {"value": "same", "unit": "pound"},
            "explanation": {"reasoning": "weight is identical"}
        }
        assert result == expected
