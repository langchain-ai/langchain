"""Test Chat Abso API wrapper."""

from typing import List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.abso import (
    ChatAbso,
)


@pytest.mark.scheduled
def test_abso_call() -> None:
    """Test valid call to abso."""
    chat = ChatAbso(fast_model="gpt-4o", slow_model="o3-mini")
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_abso_generate() -> None:
    """Test generate method of abso."""
    chat = ChatAbso(fast_model="gpt-4o", slow_model="o3-mini")
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="What's the meaning of life")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy
