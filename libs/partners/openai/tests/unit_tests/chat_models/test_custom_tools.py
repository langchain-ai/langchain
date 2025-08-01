"""Test custom tools functionality."""

from langchain_core.messages import AIMessage
from langchain_core.messages.tool import tool_call
from langchain_core.tools import tool

from langchain_openai.chat_models.base import _convert_dict_to_message


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
