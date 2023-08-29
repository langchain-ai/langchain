"""Test JinaChat wrapper."""


import pytest

from langchain.callbacks.manager import CallbackManager
from langchain.chat_models.jinachat import JinaChat
from langchain.schema import (
    BaseMessage,
    ChatGeneration,
    HumanMessage,
    LLMResult,
    SystemMessage,
)
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_jinachat() -> None:
    """Test JinaChat wrapper."""
    chat = JinaChat(max_tokens=10)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_jinachat_system_message() -> None:
    """Test JinaChat wrapper with system message."""
    chat = JinaChat(max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_jinachat_generate() -> None:
    """Test JinaChat wrapper with generate."""
    chat = JinaChat(max_tokens=10)
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


def test_jinachat_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = JinaChat(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.asyncio
async def test_async_jinachat() -> None:
    """Test async generation."""
    chat = JinaChat(max_tokens=102)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.asyncio
async def test_async_jinachat_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = JinaChat(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test_jinachat_extra_kwargs() -> None:
    """Test extra kwargs to chat openai."""
    # Check that foo is saved in extra_kwargs.
    llm = JinaChat(foo=3, max_tokens=10)
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = JinaChat(foo=3, model_kwargs={"bar": 2})
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        JinaChat(foo=3, model_kwargs={"foo": 2})

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        JinaChat(model_kwargs={"temperature": 0.2})
