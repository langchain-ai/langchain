"""Test MistralAI Chat API wrapper."""
import os
from typing import Any, AsyncGenerator, Generator
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

# TODO: Remove 'type: ignore' once mistralai has stubs or py.typed marker.
from mistralai.models.chat_completion import (  # type: ignore[import]
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
)
from mistralai.models.chat_completion import (
    ChatMessage as MistralChatMessage,
)

from langchain_mistralai.chat_models import (  # type: ignore[import]
    ChatMistralAI,
    _convert_message_to_mistral_chat_message,
)

os.environ["MISTRAL_API_KEY"] = "foo"


@pytest.mark.requires("mistralai")
def test_mistralai_model_param() -> None:
    llm = ChatMistralAI(model="foo")
    assert llm.model == "foo"


@pytest.mark.requires("mistralai")
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
            MistralChatMessage(role="system", content="Hello"),
        ),
        (
            HumanMessage(content="Hello"),
            MistralChatMessage(role="user", content="Hello"),
        ),
        (
            AIMessage(content="Hello"),
            MistralChatMessage(role="assistant", content="Hello"),
        ),
        (
            ChatMessage(role="assistant", content="Hello"),
            MistralChatMessage(role="assistant", content="Hello"),
        ),
    ],
)
def test_convert_message_to_mistral_chat_message(
    message: BaseMessage, expected: MistralChatMessage
) -> None:
    result = _convert_message_to_mistral_chat_message(message)
    assert result == expected


def _make_completion_response_from_token(token: str) -> ChatCompletionStreamResponse:
    return ChatCompletionStreamResponse(
        id="abc123",
        model="fake_model",
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=token),
                finish_reason=None,
            )
        ],
    )


def mock_chat_stream(*args: Any, **kwargs: Any) -> Generator:
    for token in ["Hello", " how", " can", " I", " help", "?"]:
        yield _make_completion_response_from_token(token)


async def mock_chat_astream(*args: Any, **kwargs: Any) -> AsyncGenerator:
    for token in ["Hello", " how", " can", " I", " help", "?"]:
        yield _make_completion_response_from_token(token)


class MyCustomHandler(BaseCallbackHandler):
    last_token: str = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.last_token = token


@patch("mistralai.client.MistralClient.chat_stream", new=mock_chat_stream)
def test_stream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatMistralAI(callbacks=[callback])
    for token in chat.stream("Hello"):
        assert callback.last_token == token.content


@patch("mistralai.async_client.MistralAsyncClient.chat_stream", new=mock_chat_astream)
async def test_astream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatMistralAI(callbacks=[callback])
    async for token in chat.astream("Hello"):
        assert callback.last_token == token.content
