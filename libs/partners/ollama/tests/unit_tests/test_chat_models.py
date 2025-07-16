"""Test chat model integration."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from httpx import Client, Request, Response
from langchain_core.messages import ChatMessage
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_ollama.chat_models import ChatOllama, _parse_arguments_from_tool_call

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
