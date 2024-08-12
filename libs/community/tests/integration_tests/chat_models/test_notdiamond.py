"""Test Not Diamond wrapper."""

from typing import List

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models import ChatNotDiamond
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_notdiamond_call() -> None:
    """Test valid call to notdiamond."""
    chat = ChatNotDiamond(llm_configs=["openai/gpt-3.5-turbo"])
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_notdiamond_call_incorrect_model() -> None:
    """Test invalid modelName"""
    message = HumanMessage(content="Hello World")
    with pytest.raises(Exception):
        chat = ChatNotDiamond(llm_configs=["openai/gpt-0"])


def test_notdiamond_generate() -> None:
    """Test generate method of anthropic."""
    chat = ChatNotDiamond(llm_configs=["openai/gpt-3.5-turbo"])
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="What is 10 * 12?")]
    ]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content


def test_notdiamond_streaming() -> None:
    """Test streaming tokens from anthropic."""
    chat = ChatNotDiamond(llm_configs=["openai/gpt-3.5-turbo"], streaming=True)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_notdiamond_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatNotDiamond(
        llm_configs=["openai/gpt-3.5-turbo"],
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")
    chat.invoke([message])
    assert callback_handler.llm_streams > 1


async def test_async_notdiamond_generate() -> None:
    """Test async generation."""
    chat = ChatNotDiamond(llm_configs=["openai/gpt-3.5-turbo"])
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="What is 10 * 12?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = await chat.agenerate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy
