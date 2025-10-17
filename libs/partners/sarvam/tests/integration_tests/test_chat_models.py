"""Integration tests for ChatSarvam.

These tests require a valid SARVAM_API_KEY environment variable.
"""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_sarvam import ChatSarvam


class TestChatSarvamIntegration(ChatModelIntegrationTests):
    """Integration tests for ChatSarvam."""

    @property
    def chat_model_class(self) -> type[ChatSarvam]:
        """Get chat model class."""
        return ChatSarvam

    @property
    def chat_model_params(self) -> dict:
        """Get chat model parameters."""
        return {"model": "sarvam-m", "temperature": 0}


@pytest.mark.scheduled
def test_chat_sarvam_invoke() -> None:
    """Test basic invoke functionality."""
    chat = ChatSarvam(model="sarvam-m")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Say 'Hello World' and nothing else."),
    ]
    response = chat.invoke(messages)
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.scheduled
def test_chat_sarvam_streaming() -> None:
    """Test streaming functionality."""
    chat = ChatSarvam(model="sarvam-m", streaming=True)
    messages = [HumanMessage(content="Count from 1 to 3")]
    chunks = []
    for chunk in chat.stream(messages):
        chunks.append(chunk.content)

    full_response = "".join(chunks)
    assert len(full_response) > 0


@pytest.mark.scheduled
def test_chat_sarvam_with_temperature() -> None:
    """Test with different temperature settings."""
    chat = ChatSarvam(model="sarvam-m", temperature=0.9)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_sarvam_with_max_tokens() -> None:
    """Test with max_tokens parameter."""
    chat = ChatSarvam(model="sarvam-m", max_tokens=50)
    message = HumanMessage(content="Tell me a long story")
    response = chat.invoke([message])
    assert isinstance(response.content, str)
