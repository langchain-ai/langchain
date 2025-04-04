"""Test Reka API wrapper."""

import logging
from typing import List

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.reka import ChatReka
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_call() -> None:
    """Test a simple call to Reka."""
    chat = ChatReka(model="reka-flash", verbose=True)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    logger.debug(f"Response content: {response.content}")


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_generate() -> None:
    """Test the generate method of Reka."""
    chat = ChatReka(model="reka-flash", verbose=True)
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
        logger.debug(f"Generated response: {response.text}")
    assert chat_messages == messages_copy


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_streaming() -> None:
    """Test streaming tokens from Reka."""
    chat = ChatReka(model="reka-flash", streaming=True, verbose=True)
    message = HumanMessage(content="Tell me a story.")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    logger.debug(f"Streaming response content: {response.content}")


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_streaming_callback() -> None:
    """Test that streaming correctly invokes callbacks."""
    callback_handler = FakeCallbackHandler()
    chat = ChatReka(
        model="reka-flash",
        streaming=True,
        callbacks=[callback_handler],
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")
    chat.invoke([message])
    assert callback_handler.llm_streams > 1
    logger.debug(f"Number of LLM streams: {callback_handler.llm_streams}")


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
async def test_reka_async_streaming_callback() -> None:
    """Test asynchronous streaming with callbacks."""
    callback_handler = FakeCallbackHandler()
    chat = ChatReka(
        model="reka-flash",
        streaming=True,
        callbacks=[callback_handler],
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
        logger.debug(f"Async generated response: {response.text}")


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_tool_usage_integration() -> None:
    """Test tool usage with Reka API integration."""
    # Initialize the ChatReka model with tools and verbose logging
    chat_reka = ChatReka(model="reka-flash", verbose=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_product_availability",
                "description": (
                    "Determine whether a product is currently in stock given "
                    "a product ID."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": (
                                "The unique product ID to check availability for"
                            ),
                        },
                    },
                    "required": ["product_id"],
                },
            },
        },
    ]
    chat_reka_with_tools = chat_reka.bind_tools(tools)

    # Start a conversation
    messages: List[BaseMessage] = [
        HumanMessage(content="Is product A12345 in stock right now?")
    ]

    # Get the initial response
    response = chat_reka_with_tools.invoke(messages)
    assert isinstance(response, AIMessage)
    logger.debug(f"Initial AI message: {response.content}")

    # Check if the model wants to use a tool
    if "tool_calls" in response.additional_kwargs:
        tool_calls = response.additional_kwargs["tool_calls"]
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            logger.debug(
                f"Tool call requested: {function_name} with arguments {arguments}"
            )

            # Simulate executing the tool
            tool_output = "AVAILABLE"

            tool_message = ToolMessage(
                content=tool_output, tool_call_id=tool_call["id"]
            )
            messages.append(response)
            messages.append(tool_message)

            final_response = chat_reka_with_tools.invoke(messages)
            assert isinstance(final_response, AIMessage)
            logger.debug(f"Final AI message: {final_response.content}")

            # Assert that the response message is non-empty
            assert final_response.content, "The final response content is empty."
    else:
        pytest.fail("The model did not request a tool.")


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_system_message() -> None:
    """Test Reka with system message."""
    chat = ChatReka(model="reka-flash", verbose=True)
    messages = [
        SystemMessage(content="You are a helpful AI that speaks like Shakespeare."),
        HumanMessage(content="Tell me about the weather today."),
    ]
    response = chat.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    logger.debug(f"Response with system message: {response.content}")


@pytest.mark.skip(
    reason="Dependency conflict w/ other dependencies for urllib3 versions."
)
def test_reka_system_message_multi_turn() -> None:
    """Test multi-turn conversation with system message."""
    chat = ChatReka(model="reka-flash", verbose=True)
    messages = [
        SystemMessage(content="You are a math tutor who explains concepts simply."),
        HumanMessage(content="What is a prime number?"),
    ]

    # First turn
    response1 = chat.invoke(messages)
    assert isinstance(response1, AIMessage)
    messages.append(response1)

    # Second turn
    messages.append(HumanMessage(content="Can you give me an example?"))
    response2 = chat.invoke(messages)
    assert isinstance(response2, AIMessage)

    logger.debug(f"First response: {response1.content}")
    logger.debug(f"Second response: {response2.content}")
