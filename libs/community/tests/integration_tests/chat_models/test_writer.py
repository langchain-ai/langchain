"""Integration tests for Writer Chat API wrapper."""
import os

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.callbacks import CallbackManager
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

from langchain_community.chat_models.writer import ChatWriter
from writerai import Writer, AsyncWriter

load_dotenv()

requires_api_key = pytest.mark.skipif(
    "os.environ.get('WRITER_API_KEY') is None",
    reason="Writer API key not found in environment"
)

@pytest.fixture
def chat() -> ChatWriter:
    """Get a ChatWriter instance for testing."""
    client = Writer(api_key=os.environ.get("WRITER_API_KEY"))
    async_client = AsyncWriter(api_key=os.environ.get("WRITER_API_KEY"))
    return ChatWriter(client=client, async_client=async_client, temperature=0)

# Basic completion tests

@requires_api_key
def test_basic_completion(chat: ChatWriter) -> None:
    """Test basic completion with real API."""
    response = chat.invoke([HumanMessage(content="Say hello!")])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

@requires_api_key
async def test_async_completion(chat: ChatWriter) -> None:
    """Test async completion with real API."""
    response = await chat.ainvoke([HumanMessage(content="Say hello!")])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

# System message tests

@requires_api_key
def test_system_message(chat: ChatWriter) -> None:
    """Test completion with system message."""
    messages = [
        SystemMessage(content="You are a helpful author of Python-inspired poetry."),
        HumanMessage(content="Write a poem about the Python programming language.")
    ]
    response = chat.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    # Should contain Python-inspired language
    assert any(word in response.content.lower() 
              for word in ["python", "programming", "language"])

# Streaming tests

@requires_api_key
def test_streaming(chat: ChatWriter) -> None:
    """Test streaming with callback handler."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    
    chat_with_streaming = ChatWriter(
        streaming=True,
        callback_manager=callback_manager,
        temperature=0
    )
    
    message = HumanMessage(content="Count to 5.")
    response = chat_with_streaming.invoke([message])
    
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert callback_handler.llm_streams > 0
    # Should contain numbers 1-5
    assert all(str(i) in response.content for i in range(1, 6))

@requires_api_key
async def test_async_streaming(chat: ChatWriter) -> None:
    """Test async streaming with callback handler."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    
    chat_with_streaming = ChatWriter(
        streaming=True,
        callback_manager=callback_manager,
        temperature=0
    )
    
    message = HumanMessage(content="Count to 5.")
    response = await chat_with_streaming.ainvoke([message])
    
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert callback_handler.llm_streams > 0
    assert all(str(i) in response.content for i in range(1, 6))

# Tool calling tests

@requires_api_key
def test_tool_calling(chat: ChatWriter) -> None:
    """Test tool calling functionality."""
    class GetWeather(BaseModel):
        """Get the weather in a location."""
        location: str = Field(..., description="The location to get weather for")
    
    chat_with_tools = chat.bind_tools(
        tools=[GetWeather],
        tool_choice="auto"
    )
    
    response = chat_with_tools.invoke(
        "What's the weather in London?"
    )
    
    assert isinstance(response, AIMessage)
    assert response.tool_calls
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "GetWeather"
    args = response.tool_calls[0].args
    assert "London" in args.lower()

@requires_api_key
async def test_async_tool_calling(chat: ChatWriter) -> None:
    """Test async tool calling functionality."""
    class GetWeather(BaseModel):
        """Get the weather in a location."""
        location: str = Field(..., description="The location to get weather for")
    
    chat_with_tools = chat.bind_tools(
        tools=[GetWeather],
        tool_choice={"type": "function", "function": {"name": "GetWeather"}}
    )
    
    response = await chat_with_tools.ainvoke(
        "What's the weather in London?"
    )
    
    assert isinstance(response, AIMessage)
    assert response.tool_calls
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "GetWeather"
    args = response.tool_calls[0].args
    assert "London" in args.lower()

# Batch operation tests

@requires_api_key
def test_batch_operations(chat: ChatWriter) -> None:
    """Test batch completion operations."""
    messages = ["Hello!", "How are you?"]
    responses = chat.batch(messages)
    
    assert len(responses) == 2
    for response in responses:
        assert isinstance(response, AIMessage)
        assert isinstance(response.content, str)
        assert len(response.content) > 0

@requires_api_key
async def test_async_batch_operations(chat: ChatWriter) -> None:
    """Test async batch completion operations."""
    messages = ["Hello!", "How are you?"]
    responses = await chat.abatch(messages)
    
    assert len(responses) == 2
    for response in responses:
        assert isinstance(response, AIMessage)
        assert isinstance(response.content, str)
        assert len(response.content) > 0

# Error handling tests

@requires_api_key
def test_invalid_api_key() -> None:
    """Test behavior with invalid API key."""
    chat = ChatWriter(writer_api_key="invalid_key")
    with pytest.raises(Exception):
        chat.invoke([HumanMessage(content="Hello!")])

@requires_api_key
def test_context_length() -> None:
    """Test handling of context length limits."""
    chat = ChatWriter()
    # Create a very long input
    long_input = "hello " * 10000
    with pytest.raises(Exception):
        chat.invoke([HumanMessage(content=long_input)])