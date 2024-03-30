"""Test MistralAI Chat API wrapper."""

import os
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

from langchain_mistralai.chat_models import (  # type: ignore[import]
    ChatMistralAI,
    _convert_message_to_mistral_chat_message,
)

os.environ["MISTRAL_API_KEY"] = "foo"


def test_mistralai_model_param() -> None:
    llm = ChatMistralAI(model="foo")
    assert llm.model == "foo"


def test_mistralai_initialization() -> None:
    """Test ChatMistralAI initialization."""
    # Verify that ChatMistralAI can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    ChatMistralAI(model="test", mistral_api_key="test")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            SystemMessage(content="Hello"),
            dict(role="system", content="Hello"),
        ),
        (
            HumanMessage(content="Hello"),
            dict(role="user", content="Hello"),
        ),
        (
            AIMessage(content="Hello"),
            dict(role="assistant", content="Hello", tool_calls=None),
        ),
        (
            ChatMessage(role="assistant", content="Hello"),
            dict(role="assistant", content="Hello"),
        ),
    ],
)
def test_convert_message_to_mistral_chat_message(
    message: BaseMessage, expected: Dict
) -> None:
    result = _convert_message_to_mistral_chat_message(message)
    assert result == expected


def _make_completion_response_from_token(token: str) -> Dict:
    return dict(
        id="abc123",
        model="fake_model",
        choices=[
            dict(
                index=0,
                delta=dict(content=token),
                finish_reason=None,
            )
        ],
    )


def mock_chat_stream(*args: Any, **kwargs: Any) -> Generator:
    def it() -> Generator:
        for token in ["Hello", " how", " can", " I", " help", "?"]:
            yield _make_completion_response_from_token(token)

    return it()


async def mock_chat_astream(*args: Any, **kwargs: Any) -> AsyncGenerator:
    async def it() -> AsyncGenerator:
        for token in ["Hello", " how", " can", " I", " help", "?"]:
            yield _make_completion_response_from_token(token)

    return it()


class MyCustomHandler(BaseCallbackHandler):
    last_token: str = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.last_token = token


@patch(
    "langchain_mistralai.chat_models.ChatMistralAI.completion_with_retry",
    new=mock_chat_stream,
)
def test_stream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatMistralAI(callbacks=[callback])
    for token in chat.stream("Hello"):
        assert callback.last_token == token.content


@patch("langchain_mistralai.chat_models.acompletion_with_retry", new=mock_chat_astream)
async def test_astream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatMistralAI(callbacks=[callback])
    async for token in chat.astream("Hello"):
        assert callback.last_token == token.content
