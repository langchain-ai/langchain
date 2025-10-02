"""Ollama specific chat model integration tests"""

from __future__ import annotations

from typing import Annotated, Optional
from unittest.mock import MagicMock, patch

import pytest
from httpx import ConnectError
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.tool import ToolCallChunk, ToolMessage
from langchain_core.tools import tool
from ollama import ResponseError
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama

DEFAULT_MODEL_NAME = "llama3.1"
REASONING_MODEL_NAME = "gpt-oss:20b"


@tool
def get_current_weather(location: str) -> dict:
    """Gets the current weather in a given location."""
    if "boston" in location.lower():
        return {"temperature": "15°F", "conditions": "snow"}
    return {"temperature": "unknown", "conditions": "unknown"}


@patch("langchain_ollama.chat_models.Client.list")
def test_init_model_not_found(mock_list: MagicMock) -> None:
    """Test that a ValueError is raised when the model is not found."""
    mock_list.side_effect = ValueError("Test model not found")
    with pytest.raises(ValueError) as excinfo:
        ChatOllama(model="non-existent-model", validate_model_on_init=True)
    assert "Test model not found" in str(excinfo.value)


@patch("langchain_ollama.chat_models.Client.list")
def test_init_connection_error(mock_list: MagicMock) -> None:
    """Test that a ValidationError is raised on connect failure during init."""
    mock_list.side_effect = ConnectError("Test connection error")

    with pytest.raises(ValidationError) as excinfo:
        ChatOllama(model="any-model", validate_model_on_init=True)
    assert "Failed to connect to Ollama" in str(excinfo.value)


@patch("langchain_ollama.chat_models.Client.list")
def test_init_response_error(mock_list: MagicMock) -> None:
    """Test that a ResponseError is raised."""
    mock_list.side_effect = ResponseError("Test response error")

    with pytest.raises(ValidationError) as excinfo:
        ChatOllama(model="any-model", validate_model_on_init=True)
    assert "Received an error from the Ollama API" in str(excinfo.value)


@pytest.mark.parametrize(("method"), [("function_calling"), ("json_schema")])
def test_structured_output(method: str) -> None:
    """Test to verify structured output via tool calling and `format` parameter."""

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    llm = ChatOllama(model=DEFAULT_MODEL_NAME, temperature=0)
    query = "Tell me a joke about cats."

    # Pydantic
    if method == "function_calling":
        structured_llm = llm.with_structured_output(Joke, method="function_calling")
        result = structured_llm.invoke(query)
        assert isinstance(result, Joke)

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, Joke)

    # JSON Schema
    if method == "json_schema":
        structured_llm = llm.with_structured_output(
            Joke.model_json_schema(), method="json_schema"
        )
        result = structured_llm.invoke(query)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)
        assert set(chunk.keys()) == {"setup", "punchline"}

        # Typed Dict
        class JokeSchema(TypedDict):
            """Joke to tell user."""

            setup: Annotated[str, "question to set up a joke"]
            punchline: Annotated[str, "answer to resolve the joke"]

        structured_llm = llm.with_structured_output(JokeSchema, method="json_schema")
        result = structured_llm.invoke(query)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)
        assert set(chunk.keys()) == {"setup", "punchline"}


@pytest.mark.parametrize(("model"), [(DEFAULT_MODEL_NAME)])
def test_structured_output_deeply_nested(model: str) -> None:
    """Test to verify structured output with a nested objects."""
    llm = ChatOllama(model=model, temperature=0)

    class Person(BaseModel):
        """Information about a person."""

        name: Optional[str] = Field(default=None, description="The name of the person")
        hair_color: Optional[str] = Field(
            default=None, description="The color of the person's hair if known"
        )
        height_in_meters: Optional[str] = Field(
            default=None, description="Height measured in meters"
        )

    class Data(BaseModel):
        """Extracted data about people."""

        people: list[Person]

    chat = llm.with_structured_output(Data)
    text = (
        "Alan Smith is 6 feet tall and has blond hair."
        "Alan Poe is 3 feet tall and has grey hair."
    )
    result = chat.invoke(text)
    assert isinstance(result, Data)

    for chunk in chat.stream(text):
        assert isinstance(chunk, Data)


@pytest.mark.parametrize(("model"), [(DEFAULT_MODEL_NAME)])
def test_tool_streaming(model: str) -> None:
    """Test that the model can stream tool calls."""
    llm = ChatOllama(model=model)
    chat_model_with_tools = llm.bind_tools([get_current_weather])

    prompt = [HumanMessage("What is the weather today in Boston?")]

    # Flags and collectors for validation
    tool_chunk_found = False
    final_tool_calls = []
    collected_tool_chunks: list[ToolCallChunk] = []

    # Stream the response and inspect the chunks
    for chunk in chat_model_with_tools.stream(prompt):
        assert isinstance(chunk, AIMessageChunk), "Expected AIMessageChunk type"

        if chunk.tool_call_chunks:
            tool_chunk_found = True
            collected_tool_chunks.extend(chunk.tool_call_chunks)

        if chunk.tool_calls:
            final_tool_calls.extend(chunk.tool_calls)

    assert tool_chunk_found, "Tool streaming did not produce any tool_call_chunks."
    assert len(final_tool_calls) == 1, (
        f"Expected 1 final tool call, but got {len(final_tool_calls)}"
    )

    final_tool_call = final_tool_calls[0]
    assert final_tool_call["name"] == "get_current_weather"
    assert final_tool_call["args"] == {"location": "Boston"}

    assert len(collected_tool_chunks) > 0
    assert collected_tool_chunks[0]["name"] == "get_current_weather"

    # The ID should be consistent across chunks that have it
    tool_call_id = collected_tool_chunks[0].get("id")
    assert tool_call_id is not None
    assert all(
        chunk.get("id") == tool_call_id
        for chunk in collected_tool_chunks
        if chunk.get("id")
    )
    assert final_tool_call["id"] == tool_call_id


@pytest.mark.parametrize(("model"), [(DEFAULT_MODEL_NAME)])
async def test_tool_astreaming(model: str) -> None:
    """Test that the model can stream tool calls."""
    llm = ChatOllama(model=model)
    chat_model_with_tools = llm.bind_tools([get_current_weather])

    prompt = [HumanMessage("What is the weather today in Boston?")]

    # Flags and collectors for validation
    tool_chunk_found = False
    final_tool_calls = []
    collected_tool_chunks: list[ToolCallChunk] = []

    # Stream the response and inspect the chunks
    async for chunk in chat_model_with_tools.astream(prompt):
        assert isinstance(chunk, AIMessageChunk), "Expected AIMessageChunk type"

        if chunk.tool_call_chunks:
            tool_chunk_found = True
            collected_tool_chunks.extend(chunk.tool_call_chunks)

        if chunk.tool_calls:
            final_tool_calls.extend(chunk.tool_calls)

    assert tool_chunk_found, "Tool streaming did not produce any tool_call_chunks."
    assert len(final_tool_calls) == 1, (
        f"Expected 1 final tool call, but got {len(final_tool_calls)}"
    )

    final_tool_call = final_tool_calls[0]
    assert final_tool_call["name"] == "get_current_weather"
    assert final_tool_call["args"] == {"location": "Boston"}

    assert len(collected_tool_chunks) > 0
    assert collected_tool_chunks[0]["name"] == "get_current_weather"

    # The ID should be consistent across chunks that have it
    tool_call_id = collected_tool_chunks[0].get("id")
    assert tool_call_id is not None
    assert all(
        chunk.get("id") == tool_call_id
        for chunk in collected_tool_chunks
        if chunk.get("id")
    )
    assert final_tool_call["id"] == tool_call_id


@pytest.mark.parametrize(
    ("model", "output_version"),
    [(REASONING_MODEL_NAME, None), (REASONING_MODEL_NAME, "v1")],
)
def test_agent_loop(model: str, output_version: Optional[str]) -> None:
    """Test agent loop with tool calling and message passing."""

    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny and 75 degrees."

    llm = ChatOllama(model=model, output_version=output_version, reasoning="low")
    llm_with_tools = llm.bind_tools([get_weather])

    input_message = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)

    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert "location" in tool_call["args"]

    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content
    assert isinstance(tool_message.content, str)
    assert "sunny" in tool_message.content.lower()

    resp_message = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    follow_up = HumanMessage("Explain why that might be using a reasoning step.")
    assert isinstance(resp_message, AIMessage)
    assert len(resp_message.content) > 0

    response = llm_with_tools.invoke(
        [input_message, tool_call_message, tool_message, resp_message, follow_up]
    )
    assert isinstance(resp_message, AIMessage)
    assert len(resp_message.content) > 0

    if output_version == "v1":
        content_blocks = response.content_blocks
        assert content_blocks is not None
        assert len(content_blocks) > 0
        assert any(block["type"] == "text" for block in content_blocks)
        assert any(block["type"] == "reasoning" for block in content_blocks)
