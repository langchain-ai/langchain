"""Test Anthropic API wrapper."""
from typing import List

import pytest

from langchain.callbacks.base import CallbackManager
from langchain.chat_models.anthropic import ChatAnthropic
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    HumanMessage,
    LLMResult,
)
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_anthropic_call() -> None:
    """Test valid call to anthropic."""
    chat = ChatAnthropic(model="test")
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_anthropic_streaming() -> None:
    """Test streaming tokens from anthropic."""
    chat = ChatAnthropic(model="test", streaming=True)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_anthropic_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatAnthropic(
        model="test",
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")
    chat([message])
    assert callback_handler.llm_streams > 1


@pytest.mark.asyncio
async def test_anthropic_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatAnthropic(
        model="test",
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    chat_messages: List[BaseMessage] = [
        HumanMessage(content="How many toes do dogs have?")
    ]
    result: LLMResult = await chat.agenerate([chat_messages])
    assert callback_handler.llm_streams > 1
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content


def test_formatting() -> None:
    chat = ChatAnthropic()

    chat_messages: List[BaseMessage] = [HumanMessage(content="Hello")]
    result = chat._convert_messages_to_prompt(chat_messages)
    assert result == "\n\nHuman: Hello\n\nAssistant:"

    chat_messages = [HumanMessage(content="Hello"), AIMessage(content="Answer:")]
    result = chat._convert_messages_to_prompt(chat_messages)
    assert result == "\n\nHuman: Hello\n\nAssistant: Answer:"
