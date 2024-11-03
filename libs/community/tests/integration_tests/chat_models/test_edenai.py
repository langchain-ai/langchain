"""Test EdenAI API wrapper."""

from typing import List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.edenai import (
    ChatEdenAI,
)


@pytest.mark.scheduled
def test_chat_edenai() -> None:
    """Test ChatEdenAI wrapper."""
    chat = ChatEdenAI(  # type: ignore[call-arg]
        provider="openai", model="gpt-3.5-turbo", temperature=0, max_tokens=1000
    )
    message = HumanMessage(content="Who are you ?")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_edenai_generate() -> None:
    """Test generate method of edenai."""
    chat = ChatEdenAI(provider="google")  # type: ignore[call-arg]
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="What is the meaning of life?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy


@pytest.mark.scheduled
async def test_edenai_async_generate() -> None:
    """Test async generation."""
    chat = ChatEdenAI(provider="google", max_tokens=50)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    result: LLMResult = await chat.agenerate([[message], [message]])
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content


@pytest.mark.scheduled
def test_edenai_streaming() -> None:
    """Test streaming EdenAI chat."""
    llm = ChatEdenAI(provider="openai", max_tokens=50)  # type: ignore[call-arg]

    for chunk in llm.stream("Generate a high fantasy story."):
        assert isinstance(chunk.content, str)


@pytest.mark.scheduled
async def test_edenai_astream() -> None:
    """Test streaming from EdenAI."""
    llm = ChatEdenAI(provider="openai", max_tokens=50)  # type: ignore[call-arg]

    async for token in llm.astream("Generate a high fantasy story."):
        assert isinstance(token.content, str)
