"""Tests for invalid_tool_calls handling in agent routing."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from langchain.agents.factory import _make_model_to_tools_edge


class TestInvalidToolCallsRouting:
    """Test that _make_model_to_tools_edge handles invalid_tool_calls correctly."""

    def _make_edge(self):
        return _make_model_to_tools_edge(
            model_destination="model",
            structured_output_tools={},
            end_destination="__end__",
        )

    def test_exits_when_no_tool_calls_and_no_invalid_tool_calls(self):
        """Agent should exit when there are no tool_calls and no invalid_tool_calls."""
        edge = self._make_edge()
        state = {
            "messages": [
                AIMessage(content="Hello!", tool_calls=[], invalid_tool_calls=[]),
            ]
        }
        result = edge(state)
        assert result == "__end__"

    def test_routes_to_model_when_invalid_tool_calls_present(self):
        """Agent should route back to model when invalid_tool_calls are present."""
        edge = self._make_edge()
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[],
                    invalid_tool_calls=[
                        {
                            "id": "call-123",
                            "name": "write_file",
                            "args": '{"file_path": "test.txt", "content": "Hello"]',
                            "error": "Expecting ',' delimiter",
                        }
                    ],
                ),
            ]
        }
        result = edge(state)
        # Should route to model, not exit
        assert result == "model"

    def test_creates_error_tool_messages_for_invalid_tool_calls(self):
        """Error ToolMessages should be appended to state for each invalid_tool_call."""
        edge = self._make_edge()
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[],
                    invalid_tool_calls=[
                        {
                            "id": "call-123",
                            "name": "write_file",
                            "args": '{"bad json"]',
                            "error": "Expecting ',' delimiter",
                        },
                        {
                            "id": "call-456",
                            "name": "read_file",
                            "args": '{"also bad"}',
                            "error": "Extra data",
                        },
                    ],
                ),
            ]
        }
        result = edge(state)
        assert result == "model"

        # Should have appended 2 error ToolMessages
        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 2
        assert tool_messages[0].tool_call_id == "call-123"
        assert tool_messages[1].tool_call_id == "call-456"
        assert tool_messages[0].status == "error"
        assert tool_messages[1].status == "error"
        assert "Expecting ',' delimiter" in tool_messages[0].content
        assert "Extra data" in tool_messages[1].content

    def test_normal_tool_calls_still_work(self):
        """Normal tool_calls should still be routed to tools via Send."""
        edge = self._make_edge()
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-789",
                            "name": "write_file",
                            "args": {"file_path": "test.txt", "content": "Hello"},
                        }
                    ],
                    invalid_tool_calls=[],
                ),
            ]
        }
        result = edge(state)
        # Should return Send objects for pending tool calls
        assert isinstance(result, list)
        assert len(result) == 1
