"""Test Anthropic API wrapper."""
from typing import List

from langchain.callbacks.manager import (
    CallbackManager,
)
from langchain.chat_models.litellm import ChatLiteLLM
from langchain.schema import (
    ChatGeneration,
    LLMResult,
)
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_litellm_call() -> None:
    """Test valid call to litellm."""
    chat = ChatLiteLLM(
        model="test",
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_litellm_generate() -> None:
    """Test generate method of anthropic."""
    chat = ChatLiteLLM(model="test")
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
    chat = ChatLiteLLM(model="test", streaming=True)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_litellm_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatLiteLLM(
        model="test",
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")
    chat([message])
    assert callback_handler.llm_streams > 1
