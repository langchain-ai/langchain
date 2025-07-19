"""Test chat model integration."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from httpx import Client, Request, Response
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import ChatMessage
from langchain_tests.unit_tests import ChatModelUnitTests

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
    raw_response = '{"model":"sample-model","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_profile_details","arguments":{"arg_1":"12345678901234567890123456"}}}]},"done":false}'  # noqa: E501
    raw_tool_calls = json.loads(raw_response)["message"]["tool_calls"]
    response = _parse_arguments_from_tool_call(raw_tool_calls[0])
    assert response is not None
    assert isinstance(response["arg_1"], str)


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


# --- Regression tests for tool-call argument parsing (see #30910) ---


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
    """Tests that _parse_json_string correctly parses valid and fixable strings."""
    raw_tool_call = {"function": {"name": "test_func", "arguments": input_string}}
    result = _parse_json_string(input_string, raw_tool_call=raw_tool_call, skip=False)
    assert result == expected_output


def test_parse_json_string_failure_case_raises_exception() -> None:
    """Tests that _parse_json_string raises an exception for truly malformed strings."""
    malformed_string = "{'key': 'value',,}"
    raw_tool_call = {"function": {"name": "test_func", "arguments": malformed_string}}
    with pytest.raises(OutputParserException):
        _parse_json_string(
            malformed_string,
            raw_tool_call=raw_tool_call,
            skip=False,
        )


def test_parse_json_string_skip_returns_input_on_failure() -> None:
    """Tests that skip=True returns the original string on parse failure."""
    malformed_string = "{'not': valid,,,}"
    raw_tool_call = {"function": {"name": "test_func", "arguments": malformed_string}}
    result = _parse_json_string(
        malformed_string,
        raw_tool_call=raw_tool_call,
        skip=True,
    )
    assert result == malformed_string
