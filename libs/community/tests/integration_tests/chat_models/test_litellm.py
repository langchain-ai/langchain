"""Test Anthropic API wrapper."""

from typing import List

from langchain_core.callbacks import (
    CallbackManager,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.litellm import ChatLiteLLM
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_litellm_call() -> None:
    """Test valid call to litellm."""
    chat = ChatLiteLLM(  # type: ignore[call-arg]
        model="test",
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_litellm_generate() -> None:
    """Test generate method of anthropic."""
    chat = ChatLiteLLM(model="test")  # type: ignore[call-arg]
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="How many toes do dogs have?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy


def test_litellm_streaming() -> None:
    """Test streaming tokens from anthropic."""
    chat = ChatLiteLLM(model="test", streaming=True)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_litellm_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatLiteLLM(  # type: ignore[call-arg]
        model="test",
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")
    chat.invoke([message])
    assert callback_handler.llm_streams > 1
