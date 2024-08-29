"""Test ChatYuan2 wrapper."""

from typing import List

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import (
    ChatGeneration,
    LLMResult,
)

from langchain_community.chat_models.yuan2 import ChatYuan2
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.mark.scheduled
def test_chat_yuan2() -> None:
    """Test ChatYuan2 wrapper."""
    chat = ChatYuan2(  # type: ignore[call-arg]
        yuan2_api_key="EMPTY",
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
        streaming=False,
    )
    messages = [
        HumanMessage(content="Hello"),
    ]
    response = chat.invoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_yuan2_system_message() -> None:
    """Test ChatYuan2 wrapper with system message."""
    chat = ChatYuan2(  # type: ignore[call-arg]
        yuan2_api_key="EMPTY",
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
        streaming=False,
    )
    messages = [
        SystemMessage(content="You are an AI assistant."),
        HumanMessage(content="Hello"),
    ]
    response = chat.invoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_yuan2_generate() -> None:
    """Test ChatYuan2 wrapper with generate."""
    chat = ChatYuan2(  # type: ignore[call-arg]
        yuan2_api_key="EMPTY",
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
        streaming=False,
    )
    messages: List = [
        HumanMessage(content="Hello"),
    ]
    response = chat.generate([messages])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    assert response.llm_output
    generation = response.generations[0]
    for gen in generation:
        assert isinstance(gen, ChatGeneration)
        assert isinstance(gen.text, str)
        assert gen.text == gen.message.content


@pytest.mark.scheduled
def test_chat_yuan2_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])

    chat = ChatYuan2(  # type: ignore[call-arg]
        yuan2_api_key="EMPTY",
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
        streaming=True,
        callbacks=callback_manager,
    )
    messages = [
        HumanMessage(content="Hello"),
    ]
    response = chat.invoke(messages)
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.asyncio
async def test_async_chat_yuan2() -> None:
    """Test async generation."""
    chat = ChatYuan2(  # type: ignore[call-arg]
        yuan2_api_key="EMPTY",
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
        streaming=False,
    )
    messages: List = [
        HumanMessage(content="Hello"),
    ]
    response = await chat.agenerate([messages])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    generations = response.generations[0]
    for generation in generations:
        assert isinstance(generation, ChatGeneration)
        assert isinstance(generation.text, str)
        assert generation.text == generation.message.content


@pytest.mark.asyncio
async def test_async_chat_yuan2_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])

    chat = ChatYuan2(  # type: ignore[call-arg]
        yuan2_api_key="EMPTY",
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
        streaming=True,
        callbacks=callback_manager,
    )
    messages: List = [
        HumanMessage(content="Hello"),
    ]
    response = await chat.agenerate([messages])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    generations = response.generations[0]
    for generation in generations:
        assert isinstance(generation, ChatGeneration)
        assert isinstance(generation.text, str)
        assert generation.text == generation.message.content
