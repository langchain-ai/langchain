"""Test MistralAI Chat API wrapper."""
import os

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

# TODO: Remove 'type: ignore' once mistralai has stubs or py.typed marker.
from mistralai.models.chat_completion import (  # type: ignore[import]
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
