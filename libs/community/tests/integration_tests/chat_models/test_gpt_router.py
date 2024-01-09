"""Test GPTRouter API wrapper."""
from typing import List

import pytest
from langchain_core.callbacks import (
    CallbackManager,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_community.chat_models.gpt_router import GPTRouter, GPTRouterModel
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_api_key_is_string() -> None:
    gpt_router = GPTRouter(
        gpt_router_api_base="https://example.com",
        gpt_router_api_key="secret-api-key",
    )
    assert isinstance(gpt_router.gpt_router_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    gpt_router = GPTRouter(
        gpt_router_api_base="https://example.com",
        gpt_router_api_key="secret-api-key",
    )
    print(gpt_router.gpt_router_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_gpt_router_call() -> None:
    """Test valid call to GPTRouter."""
    anthropic_claude = GPTRouterModel(
        name="claude-instant-1.2", provider_name="anthropic"
    )
    chat = GPTRouter(models_priority_list=[anthropic_claude])
    message = HumanMessage(content="Hello World")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_gpt_router_call_incorrect_model() -> None:
    """Test invalid modelName"""
    anthropic_claude = GPTRouterModel(
        name="model_does_not_exist", provider_name="anthropic"
    )
    chat = GPTRouter(models_priority_list=[anthropic_claude])
    message = HumanMessage(content="Hello World")
    with pytest.raises(Exception):
        chat([message])


def test_gpt_router_generate() -> None:
    """Test generate method of GPTRouter."""
    anthropic_claude = GPTRouterModel(
        name="claude-instant-1.2", provider_name="anthropic"
    )
    chat = GPTRouter(models_priority_list=[anthropic_claude])
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="If (5 + x = 18), what is x?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy


def test_gpt_router_streaming() -> None:
    """Test streaming tokens from GPTRouter."""
    anthropic_claude = GPTRouterModel(
        name="claude-instant-1.2", provider_name="anthropic"
    )
    chat = GPTRouter(models_priority_list=[anthropic_claude], streaming=True)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_gpt_router_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    anthropic_claude = GPTRouterModel(
        name="claude-instant-1.2", provider_name="anthropic"
    )
    chat = GPTRouter(
        models_priority_list=[anthropic_claude],
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Write me a 5 line poem.")
    chat([message])
    assert callback_handler.llm_streams > 1
