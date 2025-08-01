"""Test custom tools functionality."""

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import tool_call
from langchain_core.tools import tool

from langchain_openai.chat_models.base import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
)


def test_custom_tool_decorator():
    """Test that custom tools can be created with the `@tool` decorator."""

    @tool(custom=True)
    def execute_code(code: str) -> str:
        """Execute arbitrary Python code."""
        return f"Executed: {code}"

    assert execute_code.custom_tool is True

    result = execute_code.invoke({"text_input": "print('hello')"})
    assert result == "Executed: print('hello')"


def test_regular_tool_not_custom():
    """Test that regular tools are not marked as custom."""

    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: sunny"

    assert get_weather.custom_tool is False


def test_tool_call_with_text_input():
    """Test creating tool calls with `text_input`."""

    custom_call = tool_call(
        name="execute_code", text_input="print('hello world')", id="call_123"
    )

    assert custom_call["name"] == "execute_code"
    assert custom_call.get("text_input") == "print('hello world')"
    assert "args" not in custom_call
    assert custom_call["id"] == "call_123"


def test_tool_call_validation():
    """Test that `tool_call()` allows flexible creation."""

    # Should allow both args and text_input (validation happens at execution time)
    call_with_both = tool_call(
        name="test", args={"x": 1}, text_input="some text", id="call_123"
    )
    assert call_with_both["name"] == "test"
    assert call_with_both.get("args") == {"x": 1}
    assert call_with_both.get("text_input") == "some text"

    # Should allow empty args/text_input (backward compatibility)
    call_empty = tool_call(name="test", id="call_123")
    assert call_empty["name"] == "test"
    assert call_empty.get("args", {}) == {}


def test_custom_tool_call_parsing():
    """Test parsing custom tool calls from OpenAI response format."""

    # Simulate OpenAI custom tool call response
    openai_response = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "custom",
                "name": "execute_code",
                "input": "print('hello world')",
                "id": "call_abc123",
            }
        ],
    }

    # Parse the message
    message = _convert_dict_to_message(openai_response)

    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1

    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "execute_code"
    assert tool_call.get("text_input") == "print('hello world')"
    assert "args" not in tool_call  # Custom tools don't have an args field
    assert tool_call["id"] == "call_abc123"
    assert tool_call.get("type") == "tool_call"


def test_regular_tool_call_parsing_unchanged():
    """Test that regular tool call parsing still works."""

    # Simulate regular OpenAI function tool call response
    openai_response = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris", "unit": "celsius"}',
                },
                "id": "call_def456",
            }
        ],
    }

    # Parse the message
    message = _convert_dict_to_message(openai_response)

    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1

    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert tool_call.get("args") == {"location": "Paris", "unit": "celsius"}
    assert "text_input" not in tool_call
    assert tool_call["id"] == "call_def456"


def test_custom_tool_streaming_text_input():
    """Test streaming custom tool calls use `text_input` field."""
    chunk1 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "custom",
                "name": "execute_code",
                "input": "print('hello",
                "id": "call_abc123",
                "index": 0,
            }
        ],
    }

    chunk2 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "custom",
                "name": None,
                "input": " world')",
                "id": None,
                "index": 0,
            }
        ],
    }

    message_chunk1 = _convert_delta_to_message_chunk(chunk1, AIMessageChunk)
    message_chunk2 = _convert_delta_to_message_chunk(chunk2, AIMessageChunk)

    # Verify first chunk
    assert isinstance(message_chunk1, AIMessageChunk)
    assert len(message_chunk1.tool_call_chunks) == 1
    tool_call_chunk1 = message_chunk1.tool_call_chunks[0]
    assert tool_call_chunk1["name"] == "execute_code"
    assert tool_call_chunk1.get("text_input") == "print('hello"
    assert tool_call_chunk1.get("args") == ""  # Empty for custom tools
    assert tool_call_chunk1["id"] == "call_abc123"
    assert tool_call_chunk1["index"] == 0

    # Verify second chunk
    assert isinstance(message_chunk2, AIMessageChunk)
    assert len(message_chunk2.tool_call_chunks) == 1
    tool_call_chunk2 = message_chunk2.tool_call_chunks[0]
    assert tool_call_chunk2["name"] is None
    assert tool_call_chunk2.get("text_input") == " world')"
    assert tool_call_chunk2.get("args") == ""  # Empty for custom tools
    assert tool_call_chunk2["id"] is None
    assert tool_call_chunk2["index"] == 0

    # Test chunk aggregation
    combined = message_chunk1 + message_chunk2
    assert isinstance(combined, AIMessageChunk)
    assert len(combined.tool_call_chunks) == 1
    combined_chunk = combined.tool_call_chunks[0]
    assert combined_chunk["name"] == "execute_code"
    assert combined_chunk.get("text_input") == "print('hello world')"
    assert combined_chunk.get("args") == ""  # Empty for custom tools
    assert combined_chunk["id"] == "call_abc123"


def test_function_tool_streaming_args():
    """Test streaming function tool calls still use args field with JSON."""
    chunk1 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Par'},
                "id": "call_def456",
                "index": 0,
            }
        ],
    }

    chunk2 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": None, "arguments": 'is", "unit": "celsius"}'},
                "id": None,
                "index": 0,
            }
        ],
    }

    # Parse the chunks
    message_chunk1 = _convert_delta_to_message_chunk(chunk1, AIMessageChunk)
    message_chunk2 = _convert_delta_to_message_chunk(chunk2, AIMessageChunk)

    # Verify first chunk
    assert isinstance(message_chunk1, AIMessageChunk)
    assert len(message_chunk1.tool_call_chunks) == 1
    tool_call_chunk1 = message_chunk1.tool_call_chunks[0]
    assert tool_call_chunk1["name"] == "get_weather"
    assert tool_call_chunk1.get("args") == '{"location": "Par'
    assert "text_input" not in tool_call_chunk1
    assert tool_call_chunk1["id"] == "call_def456"
    assert tool_call_chunk1["index"] == 0

    # Verify second chunk
    assert isinstance(message_chunk2, AIMessageChunk)
    assert len(message_chunk2.tool_call_chunks) == 1
    tool_call_chunk2 = message_chunk2.tool_call_chunks[0]
    assert tool_call_chunk2["name"] is None
    assert tool_call_chunk2.get("args") == 'is", "unit": "celsius"}'
    assert "text_input" not in tool_call_chunk2
    assert tool_call_chunk2["id"] is None
    assert tool_call_chunk2["index"] == 0

    # Test chunk aggregation
    combined = message_chunk1 + message_chunk2
    assert isinstance(combined, AIMessageChunk)
    assert len(combined.tool_call_chunks) == 1
    combined_chunk = combined.tool_call_chunks[0]
    assert combined_chunk["name"] == "get_weather"
    assert combined_chunk.get("args") == '{"location": "Paris", "unit": "celsius"}'
    assert "text_input" not in combined_chunk
    assert combined_chunk["id"] == "call_def456"


def test_mixed_tool_streaming():
    """Test streaming with both custom and function tools in same response."""
    # Simulate mixed tool streaming chunk from OpenAI
    chunk = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "custom",
                "name": "execute_code",
                "input": "x = 5",
                "id": "call_custom_123",
                "index": 0,
            },
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                "id": "call_func_456",
                "index": 1,
            },
        ],
    }

    # Parse the chunk
    message_chunk = _convert_delta_to_message_chunk(chunk, AIMessageChunk)

    assert isinstance(message_chunk, AIMessageChunk)
    assert len(message_chunk.tool_call_chunks) == 2

    # Verify custom tool chunk
    custom_chunk = message_chunk.tool_call_chunks[0]
    assert custom_chunk["name"] == "execute_code"
    assert custom_chunk.get("text_input") == "x = 5"
    assert custom_chunk.get("args") == ""  # Empty for custom tools
    assert custom_chunk["id"] == "call_custom_123"
    assert custom_chunk["index"] == 0

    # Verify function tool chunk
    function_chunk = message_chunk.tool_call_chunks[1]
    assert function_chunk["name"] == "get_weather"
    assert function_chunk.get("args") == '{"location": "NYC"}'
    assert "text_input" not in function_chunk
    assert function_chunk["id"] == "call_func_456"
    assert function_chunk["index"] == 1
