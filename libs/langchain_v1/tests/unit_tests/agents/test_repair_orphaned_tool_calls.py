"""Unit tests for _repair_orphaned_tool_calls in factory.py."""

from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

from langchain.agents.factory import _repair_orphaned_tool_calls


def test_no_repair_needed() -> None:
    """Test that messages without orphaned tool calls are returned unchanged."""
    messages = [
        HumanMessage("Hello"),
        AIMessage(
            content="",
            tool_calls=[ToolCall(name="search", args={"q": "test"}, id="tc1")],
        ),
        ToolMessage(content="result", tool_call_id="tc1", name="search"),
        AIMessage(content="Done"),
    ]
    result = _repair_orphaned_tool_calls(messages)
    assert result is messages, "Should return original list when no repairs needed"


def test_repair_single_orphaned_tool_call() -> None:
    """Test that a single orphaned tool call gets a placeholder ToolMessage."""
    messages = [
        HumanMessage("Hello"),
        AIMessage(
            content="",
            tool_calls=[ToolCall(name="search", args={"q": "test"}, id="tc1")],
        ),
        # No ToolMessage for tc1
        AIMessage(content="Limit reached"),
    ]
    result = _repair_orphaned_tool_calls(messages)
    assert len(result) == 4, "Should inject one placeholder ToolMessage"

    injected = result[2]
    assert isinstance(injected, ToolMessage)
    assert injected.tool_call_id == "tc1"
    assert injected.status == "error"
    assert "search" in injected.content


def test_repair_multiple_orphaned_tool_calls() -> None:
    """Test that multiple orphaned tool calls all get placeholder ToolMessages."""
    messages = [
        HumanMessage("Hello"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="search", args={"q": "q1"}, id="tc1"),
                ToolCall(name="search", args={"q": "q2"}, id="tc2"),
                ToolCall(name="calc", args={"x": "1"}, id="tc3"),
            ],
        ),
        # No ToolMessages at all
        AIMessage(content="Limit reached"),
    ]
    result = _repair_orphaned_tool_calls(messages)

    tool_messages = [m for m in result if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 3, "All 3 orphaned tool calls need placeholders"

    repaired_ids = {m.tool_call_id for m in tool_messages}
    assert repaired_ids == {"tc1", "tc2", "tc3"}


def test_repair_partial_orphaned_tool_calls() -> None:
    """Test that only orphaned calls get repaired, not already-answered ones."""
    messages = [
        HumanMessage("Hello"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="search", args={"q": "q1"}, id="tc1"),
                ToolCall(name="search", args={"q": "q2"}, id="tc2"),
            ],
        ),
        ToolMessage(content="result1", tool_call_id="tc1", name="search"),
        # tc2 is orphaned
        AIMessage(content="Limit reached"),
    ]
    result = _repair_orphaned_tool_calls(messages)

    tool_messages = [m for m in result if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2, "Should have original + 1 injected"

    injected = [m for m in tool_messages if m.tool_call_id == "tc2"]
    assert len(injected) == 1
    assert injected[0].status == "error"


def test_no_ai_messages_with_tool_calls() -> None:
    """Test that messages without any tool calls need no repair."""
    messages = [
        HumanMessage("Hello"),
        AIMessage(content="Hi there"),
    ]
    result = _repair_orphaned_tool_calls(messages)
    assert result is messages


def test_empty_messages() -> None:
    """Test that empty message list is handled."""
    result = _repair_orphaned_tool_calls([])
    assert result == []
