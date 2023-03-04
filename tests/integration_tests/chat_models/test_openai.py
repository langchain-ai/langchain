"""Test ChatOpenAI wrapper."""

import pytest

from langchain.callbacks.base import CallbackManager
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import (
    BaseMessage,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_chat_openai():
    """Test ChatOpenAI wrapper."""
    chat = ChatOpenAI(max_tokens=10)
    message = HumanMessage(text="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.text, str)


def test_chat_openai_system_message():
    """Test ChatOpenAI wrapper with system message."""
    chat = ChatOpenAI(max_tokens=10)
    system_message = SystemMessage(text="You are to chat with the user.")
    human_message = HumanMessage(text="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.text, str)


# def test_chat_openai_generate():
#     """Test ChatOpenAI wrapper with generate."""
#     chat = ChatOpenAI(max_tokens=10)
#     message = HumanMessage(text="Hello")
#     response = chat.generate([[message]])
#     assert isinstance(response, ChatMessage)
#     assert isinstance(response.text, str)


def test_chat_openai_multiple_completions():
    """Test ChatOpenAI wrapper with multiple completions."""
    chat = ChatOpenAI(max_tokens=10, n=5)
    message = HumanMessage(text="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.text, str)


def test_chat_openai_streaming():
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatOpenAI(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(text="Hello")
    response = chat([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


def test_chat_openai_invalid_streaming_params():
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        ChatOpenAI(
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )
