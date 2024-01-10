"""Test MistralAI Chat API wrapper."""
import os
from typing import Generator

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGenerationChunk

# TODO: Remove 'type: ignore' once mistralai has stubs or py.typed marker.
from mistralai.models.chat_completion import (  # type: ignore[import]
    ChatMessage as MistralChatMessage,
)

from langchain_mistralai.chat_models import (  # type: ignore[import]
    ChatMistralAI,
    _convert_message_to_mistral_chat_message,
    _enforce_stop_tokens,
    _enforce_stop_tokens_stream,
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


def test_enforce_stop_tokens() -> None:
    """Test _enforce_stop_tokens helper function."""
    assert _enforce_stop_tokens("Hello", ["lo"]) == "Hel"
    assert _enforce_stop_tokens("Hello there my friend", ["my"]) == "Hello there "
    assert _enforce_stop_tokens("Hello there my friend", ["my", "there"]) == "Hello "

    # Test regex special characters
    assert (
        _enforce_stop_tokens("Hello there? my friend", ["e?", "friend"]) == "Hello ther"
    )


def _string_to_2_char_stream(string: str) -> Generator[ChatGenerationChunk, None, None]:
    """Convert a string to a stream of 2 character chunks."""
    for i in range(0, len(string), 2):
        message = AIMessageChunk(content=string[i : i + 2])
        yield ChatGenerationChunk(message=message)


def _stream_to_string(stream: Generator[ChatGenerationChunk, None, None]) -> str:
    """Convert a stream of chunks to a string."""
    return "".join([chunk.message.content for chunk in stream])


def test_enforce_stop_tokens_stream() -> None:
    """Test _enforce_stop_tokens_stream helper function."""
    assert (
        _stream_to_string(
            _enforce_stop_tokens_stream(_string_to_2_char_stream("Hello"), ["lo"])
        )
        == "Hel"
    )
    assert (
        _stream_to_string(
            _enforce_stop_tokens_stream(
                _string_to_2_char_stream("Hello there my friend"), ["my"]
            )
        )
        == "Hello there "
    )
    assert (
        _stream_to_string(
            _enforce_stop_tokens_stream(
                _string_to_2_char_stream("Hello there my friend"), ["my", "there"]
            )
        )
        == "Hello "
    )
