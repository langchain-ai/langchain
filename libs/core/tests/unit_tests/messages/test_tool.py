"""Tests for langchain_core.messages.tool utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import tool_messages_from_invalid_tool_calls

if TYPE_CHECKING:
    from langchain_core.messages.content import InvalidToolCall


class TestToolMessagesFromInvalidToolCalls:
    """Tests for tool_messages_from_invalid_tool_calls utility."""

    def test_basic_conversion(self) -> None:
        """Single invalid tool call produces correct error ToolMessage."""
        invalid_calls: list[InvalidToolCall] = [
            {
                "type": "invalid_tool_call",
                "name": "get_weather",
                "args": "{bad json",
                "id": "call_1",
                "error": "JSON parse error",
            }
        ]
        result = tool_messages_from_invalid_tool_calls(invalid_calls)

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, ToolMessage)
        assert msg.status == "error"
        assert msg.tool_call_id == "call_1"
        assert msg.name == "get_weather"
        assert "get_weather" in msg.content
        assert "JSON parse error" in msg.content
        assert "Please fix your mistakes" in msg.content

    def test_empty_list_returns_empty(self) -> None:
        """Empty input produces empty output."""
        result = tool_messages_from_invalid_tool_calls([])
        assert result == []

    def test_none_id_defaults_to_empty_string(self) -> None:
        """InvalidToolCall with id=None produces tool_call_id=''."""
        invalid_calls: list[InvalidToolCall] = [
            {
                "type": "invalid_tool_call",
                "name": "my_tool",
                "args": "bad",
                "id": None,
                "error": None,
            }
        ]
        result = tool_messages_from_invalid_tool_calls(invalid_calls)

        assert len(result) == 1
        assert result[0].tool_call_id == ""

    def test_none_name_defaults_to_unknown(self) -> None:
        """InvalidToolCall with name=None produces name='unknown'."""
        invalid_calls: list[InvalidToolCall] = [
            {
                "type": "invalid_tool_call",
                "name": None,
                "args": "bad",
                "id": "call_1",
                "error": None,
            }
        ]
        result = tool_messages_from_invalid_tool_calls(invalid_calls)

        assert len(result) == 1
        assert result[0].name == "unknown"
        assert "unknown" in result[0].content

    def test_none_error_omits_details_line(self) -> None:
        """When error is None, the 'Details:' line should not appear."""
        invalid_calls: list[InvalidToolCall] = [
            {
                "type": "invalid_tool_call",
                "name": "my_tool",
                "args": "bad",
                "id": "call_1",
                "error": None,
            }
        ]
        result = tool_messages_from_invalid_tool_calls(invalid_calls)

        assert len(result) == 1
        assert "Details:" not in result[0].content

    def test_multiple_invalid_tool_calls(self) -> None:
        """Multiple invalid tool calls each produce an error ToolMessage."""
        invalid_calls: list[InvalidToolCall] = [
            {
                "type": "invalid_tool_call",
                "name": "tool_a",
                "args": "{bad",
                "id": "call_1",
                "error": "parse error 1",
            },
            {
                "type": "invalid_tool_call",
                "name": "tool_b",
                "args": "{also bad",
                "id": "call_2",
                "error": "parse error 2",
            },
        ]
        result = tool_messages_from_invalid_tool_calls(invalid_calls)

        assert len(result) == 2
        assert result[0].tool_call_id == "call_1"
        assert result[0].name == "tool_a"
        assert "tool_a" in result[0].content
        assert result[1].tool_call_id == "call_2"
        assert result[1].name == "tool_b"
        assert "tool_b" in result[1].content
        assert all(msg.status == "error" for msg in result)
