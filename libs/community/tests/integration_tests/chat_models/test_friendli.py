"""Test Friendli chat API."""

import pytest
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.outputs.generation import Generation
from langchain_core.outputs.llm_result import LLMResult

from langchain_community.chat_models.friendli import ChatFriendli


@pytest.fixture
def friendli_chat() -> ChatFriendli:
    """Friendli LLM for chat."""
    return ChatFriendli(temperature=0, max_tokens=10)


def test_friendli_invoke(friendli_chat: ChatFriendli) -> None:
    """Test invoke."""
    output = friendli_chat.invoke("What is generative AI?")
    assert isinstance(output, AIMessage)
    assert isinstance(output.content, str)


async def test_friendli_ainvoke(friendli_chat: ChatFriendli) -> None:
    """Test async invoke."""
    output = await friendli_chat.ainvoke("What is generative AI?")
    assert isinstance(output, AIMessage)
    assert isinstance(output.content, str)


def test_friendli_batch(friendli_chat: ChatFriendli) -> None:
    """Test batch."""
    outputs = friendli_chat.batch(["What is generative AI?", "What is generative AI?"])
    for output in outputs:
        assert isinstance(output, AIMessage)
        assert isinstance(output.content, str)


async def test_friendli_abatch(friendli_chat: ChatFriendli) -> None:
    """Test async batch."""
    outputs = await friendli_chat.abatch(
        ["What is generative AI?", "What is generative AI?"]
    )
    for output in outputs:
        assert isinstance(output, AIMessage)
        assert isinstance(output.content, str)


def test_friendli_generate(friendli_chat: ChatFriendli) -> None:
    """Test generate."""
    message = HumanMessage(content="What is generative AI?")
    result = friendli_chat.generate([[message], [message]])
    assert isinstance(result, LLMResult)
    generations = result.generations
    assert len(generations) == 2
    for generation in generations:
        gen_ = generation[0]
        assert isinstance(gen_, Generation)
        text = gen_.text
        assert isinstance(text, str)
        generation_info = gen_.generation_info
        if generation_info is not None:
            assert "token" in generation_info


async def test_friendli_agenerate(friendli_chat: ChatFriendli) -> None:
    """Test async generate."""
    message = HumanMessage(content="What is generative AI?")
    result = await friendli_chat.agenerate([[message], [message]])
    assert isinstance(result, LLMResult)
    generations = result.generations
    assert len(generations) == 2
    for generation in generations:
        gen_ = generation[0]
        assert isinstance(gen_, Generation)
        text = gen_.text
        assert isinstance(text, str)
        generation_info = gen_.generation_info
        if generation_info is not None:
            assert "token" in generation_info


def test_friendli_stream(friendli_chat: ChatFriendli) -> None:
    """Test stream."""
    stream = friendli_chat.stream("Say hello world.")
    for chunk in stream:
        assert isinstance(chunk, AIMessage)
        assert isinstance(chunk.content, str)


async def test_friendli_astream(friendli_chat: ChatFriendli) -> None:
    """Test async stream."""
    stream = friendli_chat.astream("Say hello world.")
    async for chunk in stream:
        assert isinstance(chunk, AIMessage)
        assert isinstance(chunk.content, str)
