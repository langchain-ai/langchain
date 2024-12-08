"""Integration tests for Nebius AI Studio chat model."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.chat_models.nebius_ai_studio import ChatNebiusAIStudio


@pytest.mark.requires("openai")
def test_chat_nebius_ai_studio_call() -> None:
    """Test valid call to Nebius AI Studio chat model."""
    chat = ChatNebiusAIStudio(temperature=0.0)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Say foo:"),
    ]
    output = chat(messages)
    assert isinstance(output.content, str)


@pytest.mark.requires("openai")
def test_chat_nebius_ai_studio_streaming() -> None:
    """Test streaming from Nebius AI Studio chat model."""
    chat = ChatNebiusAIStudio(temperature=0.0, streaming=True)
    messages = [HumanMessage(content="Say foo:")]
    for chunk in chat.stream(messages):
        assert chunk.content is not None
