"""Tests for find_safe_message_cutoff."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import find_safe_message_cutoff


def test_find_safe_message_cutoff_preserves_tool_pair() -> None:
    """Cutoff must not split AIMessage tool calls from ToolMessage responses."""
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "1", "name": "search", "args": {}},
            ],
        ),
        ToolMessage(content="result", tool_call_id="1", name="search"),
        HumanMessage(content="thanks"),
    ]
    cutoff = find_safe_message_cutoff(messages, keep=1)
    assert cutoff == 2
    assert messages[cutoff:] == [HumanMessage(content="thanks")]


def test_find_safe_message_cutoff_returns_zero_when_short() -> None:
    messages = [HumanMessage(content="hi")]
    assert find_safe_message_cutoff(messages, keep=5) == 0
