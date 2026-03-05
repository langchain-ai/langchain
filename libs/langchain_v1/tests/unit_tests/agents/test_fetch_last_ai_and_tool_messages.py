"""Unit tests for _fetch_last_ai_and_tool_messages helper function.

These tests verify that the helper function correctly handles edge cases,
including the scenario where no AIMessage exists in the message list
(fixes issue #34792).
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain.agents.factory import _fetch_last_ai_and_tool_messages


def test_fetch_last_ai_and_tool_messages_normal() -> None:
    """Test normal case with AIMessage and subsequent ToolMessages."""
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!", tool_calls=[{"name": "test", "id": "1", "args": {}}]),
        ToolMessage(content="Tool result", tool_call_id="1"),
    ]

    ai_msg, tool_msgs = _fetch_last_ai_and_tool_messages(messages)

    assert ai_msg is not None
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "Hi there!"
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "Tool result"


def test_fetch_last_ai_and_tool_messages_multiple_ai() -> None:
    """Test that the last AIMessage is returned when multiple exist."""
    messages = [
        HumanMessage(content="First question"),
        AIMessage(content="First answer", id="ai1"),
        HumanMessage(content="Second question"),
        AIMessage(content="Second answer", id="ai2"),
    ]

    ai_msg, tool_msgs = _fetch_last_ai_and_tool_messages(messages)

    assert ai_msg is not None
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "Second answer"
    assert ai_msg.id == "ai2"
    assert len(tool_msgs) == 0


def test_fetch_last_ai_and_tool_messages_no_ai_message() -> None:
    """Test handling when no AIMessage exists in messages.

    This is the edge case that caused issue #34792 - UnboundLocalError
    when using RemoveMessage(id=REMOVE_ALL_MESSAGES) to clear thread messages.
    The function now returns None for the AIMessage, allowing callers to
    handle this edge case explicitly.
    """
    messages = [
        HumanMessage(content="Hello"),
        SystemMessage(content="You are a helpful assistant"),
    ]

    ai_msg, tool_msgs = _fetch_last_ai_and_tool_messages(messages)

    # Should return None when no AIMessage is found
    assert ai_msg is None
    assert len(tool_msgs) == 0


def test_fetch_last_ai_and_tool_messages_empty_list() -> None:
    """Test handling of empty messages list.

    This can occur after RemoveMessage(id=REMOVE_ALL_MESSAGES) clears all messages.
    """
    messages: list = []

    ai_msg, tool_msgs = _fetch_last_ai_and_tool_messages(messages)

    # Should return None when no AIMessage is found
    assert ai_msg is None
    assert len(tool_msgs) == 0


def test_fetch_last_ai_and_tool_messages_only_human_messages() -> None:
    """Test handling when only HumanMessages exist."""
    messages = [
        HumanMessage(content="Hello"),
        HumanMessage(content="Are you there?"),
    ]

    ai_msg, tool_msgs = _fetch_last_ai_and_tool_messages(messages)

    assert ai_msg is None
    assert len(tool_msgs) == 0


def test_fetch_last_ai_and_tool_messages_ai_without_tool_calls() -> None:
    """Test AIMessage without tool_calls returns empty tool messages list."""
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi! How can I help you today?"),
    ]

    ai_msg, tool_msgs = _fetch_last_ai_and_tool_messages(messages)

    assert ai_msg is not None
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "Hi! How can I help you today?"
    assert len(tool_msgs) == 0
