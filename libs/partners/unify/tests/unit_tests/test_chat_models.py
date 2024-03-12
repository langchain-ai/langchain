"""Test chat model integration."""

import json
from typing import Any, AsyncGenerator, Dict, Generator
from unittest.mock import patch

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_unify.chat_models import ChatUnify


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatUnify()


def test_unify_initialization() -> None:
    """Test ChatUnify initialization."""
    # Verify that ChatUnify can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    ChatUnify(model="test", mistral_api_key="test")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            [SystemMessage(content="Hello")],
            [{"role": "system", "content": "Hello"}],
        ),
        (
            [HumanMessage(content="Hello")],
            [{"role": "user", "content": "Hello"}],
        ),
        (
            [AIMessage(content="Hello")],
            [{"role": "assistant", "content": "Hello"}],
        ),
        (
            [ChatMessage(role="assistant", content="Hello")],
            [{"role": "assistant", "content": "Hello"}],
        ),
    ],
)
def test_convert_to_message(message: BaseMessage, expected: Dict) -> None:
    result = ChatUnify()._format_messages(message)
    assert result == expected


def _make_completion_response_from_token(token: str) -> Dict:
    out = {
        "id": "abc123",
        "model": "fake_model",
        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
    }
    return f"data: {json.dumps(out)}\n\n"


def mock_chat_stream(*args: Any, **kwargs: Any) -> Generator:
    for token in ["Hello", " how", " can", " I", " help", "?"]:
        yield _make_completion_response_from_token(token)


async def mock_chat_astream(*args: Any, **kwargs: Any) -> AsyncGenerator:
    async for token in ["Hello", " how", " can", " I", " help", "?"]:
        yield _make_completion_response_from_token(token)


class MyCustomHandler(BaseCallbackHandler):
    last_token: str = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.last_token = token


class MockStream:
    def __init__(self, *args, **kwargs):
        self.tokens = ["Hello", " how", " can", " I", " help", "?"]
        self.status_code = 200

    def iter_lines(self):
        for token in self.tokens:
            yield _make_completion_response_from_token(token)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass


class MockAStream(MockStream):
    async def aiter_lines(self):
        for token in self.tokens:
            yield _make_completion_response_from_token(token)

    async def __aexit__(self, type, value, traceback):
        pass

    async def __aenter__(self):
        return self


@patch("httpx.Client.stream", new=MockStream)
def test_stream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatUnify(callbacks=[callback])
    for token in chat.stream("Hello"):
        assert callback.last_token == token.content


@patch("httpx.AsyncClient.stream", new=MockAStream)
async def test_astream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatUnify(callbacks=[callback])
    async for token in chat.astream("Hello"):
        assert callback.last_token == token.content
