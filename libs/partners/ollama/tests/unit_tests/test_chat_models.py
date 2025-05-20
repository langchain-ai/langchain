"""Test chat model integration."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest
from httpx import Client, Request, Response
from langchain_core.messages import ChatMessage
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_ollama.chat_models import ChatOllama, _parse_arguments_from_tool_call


class TestChatOllama(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": "llama3-groq-tool-use"}


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
        base_url="http://whocares:11434",
        model="granite3.2",
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

def test_custom_headers_are_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_headers = {}

    # Mock the Client.stream method to capture headers
    def mock_stream(self, request: Request, *args, **kwargs):
        nonlocal captured_headers
        captured_headers = dict(request.headers)
        return _mock_httpx_client_stream()

    monkeypatch.setattr(Client, "stream", mock_stream)

    headers = {
        "Authorization": "Bearer test-token",
        "X-Custom-Header": "LangChainTest"
    }

    llm = ChatOllama(
        base_url="http://whocares:11434",
        model="granite3.2",
        headers=headers,
    )

    messages = [ChatMessage(role="user", content="Hello world")]
    llm.invoke(messages)

    assert captured_headers.get("Authorization") == "Bearer test-token"
    assert captured_headers.get("X-Custom-Header") == "LangChainTest"
