"""Test ERNIE Bot wrapper."""

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    LLMResult,
)

from langchain_community.chat_models import ErnieBotChat


def test_erniebot_call() -> None:
    """Test valid call."""
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_erniebot_generate() -> None:
    """Test generation."""
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.asyncio
async def test_erniebot_agenerate() -> None:
    """Test asynchronous generation."""
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test_erniebot_stream() -> None:
    """Test streaming."""
    chat = ErnieBotChat()
    for chunk in chat.stream("Write a joke."):
        assert isinstance(chunk.content, str)


@pytest.mark.asyncio
async def test_erniebot_astream() -> None:
    """Test asynchronous streaming."""
    chat = ErnieBotChat()
    async for chunk in chat.astream("Write a joke."):
        assert isinstance(chunk.content, str)


def test_erniebot_params() -> None:
    """Test setting parameters."""
    chat = ErnieBotChat(model="ernie-turbo", temperature=0.7)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_erniebot_chat_history() -> None:
    """Test that multiple messages works."""
    chat = ErnieBotChat()
    response = chat(
        messages=[
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
