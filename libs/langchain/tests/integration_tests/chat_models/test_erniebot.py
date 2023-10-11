"""Test ERNIE Bot wrapper."""

from typing import List

import pytest

from langchain.chat_models.erniebot import ErnieBotChat
from langchain.schema import (
    ChatGeneration,
    LLMResult,
)
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)


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
    chat = ErnieBotChat(model="ernie-bot-turbo", temperature=0.7)
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


def test_erniebot_function_calling() -> None:
    """Test function calling."""
    functions = [
        {
            "name": "get_current_temperature",
            "description": "Get the current temperature in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location", "unit"],
            },
            "responses": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "integer",
                        "description": "Temperature in the location.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
            },
        },
    ]

    chat = ErnieBotChat()
    messages: List[BaseMessage] = [
        HumanMessage(
            content="What is the temperature in Shenzhen today in degrees Celsius?"
        )
    ]

    response = chat.generate([messages], functions=functions)
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    generations = response.generations[0]
    assert len(generations) == 1
    generation = generations[0]
    assert isinstance(generation, ChatGeneration)
    assert generation.message.content == ""
    assert "function_call" in generation.message.additional_kwargs
    function_call = generation.message.additional_kwargs["function_call"]
    assert function_call["name"] == "get_current_temperature"

    messages.append(generation.message)
    messages.append(
        FunctionMessage(
            name="get_current_temperature",
            content='{"temperature":25,"unit":"celsius"}',
        )
    )

    response = chat.generate([messages])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    generations = response.generations[0]
    assert len(generations) == 1
    generation = generations[0]
    assert isinstance(generation, ChatGeneration)
    assert generation.message.content != ""
    assert "function_call" not in generation.message.additional_kwargs
