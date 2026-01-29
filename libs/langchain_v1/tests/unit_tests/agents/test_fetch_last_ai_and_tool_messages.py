"""Unit tests for _fetch_last_ai_and_tool_messages bug fix.

This test module validates the fix for the UnboundLocalError that occurred
when clearing conversation state using RemoveMessage(id=REMOVE_ALL_MESSAGES).

Bug: https://github.com/langchain-ai/langchain/issues/XXXXX
Fix: Initialize last_ai_index to -1 and handle the case where no AIMessage exists.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain.agents.factory import _fetch_last_ai_and_tool_messages


class TestFetchLastAIAndToolMessages:
    """Test suite for _fetch_last_ai_and_tool_messages function."""

    def test_empty_message_list(self):
        """Test with an empty message list.

        This is the core bug scenario - should not raise UnboundLocalError.
        Should return None to signal no AIMessage was found.
        """
        messages = []
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert ai_message is None
        assert tool_messages == []

    def test_no_ai_message_in_list(self):
        """Test with messages but no AIMessage present.

        This simulates the scenario after RemoveMessage(id=REMOVE_ALL_MESSAGES)
        when only a HumanMessage is added back.
        """
        messages = [
            HumanMessage(content="Fresh start"),
        ]
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert ai_message is None
        assert tool_messages == []

    def test_only_system_and_human_messages(self):
        """Test with system and human messages but no AIMessage.

        This is another edge case that could occur in practice.
        """
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
        ]
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert ai_message is None
        assert tool_messages == []

    def test_single_ai_message(self):
        """Test with a single AIMessage (normal case)."""
        messages = [
            AIMessage(content="Hello! How can I help you?"),
        ]
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert isinstance(ai_message, AIMessage)
        assert ai_message.content == "Hello! How can I help you?"
        assert tool_messages == []

    def test_ai_message_with_tool_calls(self):
        """Test AIMessage with tool calls but no ToolMessages yet."""
        messages = [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_123", "name": "get_weather", "args": {"city": "San Francisco"}}
                ],
            ),
        ]
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert isinstance(ai_message, AIMessage)
        assert len(ai_message.tool_calls) == 1
        assert ai_message.tool_calls[0]["name"] == "get_weather"
        assert tool_messages == []

    def test_ai_message_followed_by_tool_messages(self):
        """Test AIMessage followed by ToolMessages (normal agent flow)."""
        messages = [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_123", "name": "get_weather", "args": {"city": "San Francisco"}}
                ],
            ),
            ToolMessage(content="Sunny, 22°C", tool_call_id="call_123", name="get_weather"),
        ]
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert isinstance(ai_message, AIMessage)
        assert len(ai_message.tool_calls) == 1
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "Sunny, 22°C"
        assert tool_messages[0].tool_call_id == "call_123"

    def test_multiple_ai_messages_returns_last(self):
        """Test that only the LAST AIMessage is returned."""
        messages = [
            HumanMessage(content="First question"),
            AIMessage(content="First response"),
            HumanMessage(content="Second question"),
            AIMessage(content="Second response"),
        ]
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert isinstance(ai_message, AIMessage)
        assert ai_message.content == "Second response"
        assert tool_messages == []

    def test_multiple_tool_messages_after_ai(self):
        """Test multiple ToolMessages after an AIMessage."""
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "tool1", "args": {}},
                    {"id": "call_2", "name": "tool2", "args": {}},
                ],
            ),
            ToolMessage(content="Result 1", tool_call_id="call_1", name="tool1"),
            ToolMessage(content="Result 2", tool_call_id="call_2", name="tool2"),
        ]
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert isinstance(ai_message, AIMessage)
        assert len(tool_messages) == 2
        assert tool_messages[0].content == "Result 1"
        assert tool_messages[1].content == "Result 2"

    def test_tool_messages_before_ai_not_included(self):
        """Test that ToolMessages BEFORE the last AIMessage are not included."""
        messages = [
            AIMessage(
                content="First AI", tool_calls=[{"id": "call_old", "name": "old_tool", "args": {}}]
            ),
            ToolMessage(content="Old result", tool_call_id="call_old", name="old_tool"),
            HumanMessage(content="Another question"),
            AIMessage(content="Second AI"),  # This is the last AI message
        ]
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert isinstance(ai_message, AIMessage)
        assert ai_message.content == "Second AI"
        assert tool_messages == []  # Old ToolMessage should not be included

    def test_mixed_messages_after_ai(self):
        """Test that only ToolMessages are extracted after AIMessage."""
        messages = [
            AIMessage(content="", tool_calls=[{"id": "call_1", "name": "tool1", "args": {}}]),
            ToolMessage(content="Tool result", tool_call_id="call_1", name="tool1"),
            HumanMessage(content="Follow-up question"),  # Should be ignored
        ]
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert isinstance(ai_message, AIMessage)
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "Tool result"


class TestBugReproduction:
    """Integration-style tests that reproduce the original bug scenario."""

    def test_clear_conversation_scenario(self):
        """Reproduce the exact bug scenario from the issue.

        Scenario: After using RemoveMessage(id=REMOVE_ALL_MESSAGES),
        the state may have no AIMessage, only a fresh HumanMessage.

        Before the fix, this would raise:
            UnboundLocalError: cannot access local variable 'last_ai_index'
            where it is not associated with a value

        After the fix: Returns None to signal no AIMessage exists,
        allowing callers to handle this edge case appropriately.
        """
        # Simulate state after clearing all messages and adding a fresh start
        messages = [
            HumanMessage(content="Fresh start"),
        ]

        # This should NOT raise UnboundLocalError
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert ai_message is None
        assert tool_messages == []

    def test_clear_conversation_with_only_system_prompt(self):
        """Test clearing conversation when only system prompt remains.

        In create_agent, the system prompt is prepended dynamically,
        so after clearing, only the system message might exist.
        """
        messages = [
            SystemMessage(content="You are a helpful assistant."),
        ]

        # This should NOT raise UnboundLocalError
        ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)

        assert ai_message is None
        assert tool_messages == []


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
