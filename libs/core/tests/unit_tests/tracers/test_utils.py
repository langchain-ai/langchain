"""Unit tests for tracer utility functions."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tracers.utils import (
    count_tool_calls_in_run,
    store_tool_call_count_in_run,
)


def create_mock_run(
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock Run object for testing."""
    run = MagicMock()
    run.inputs = inputs or {}
    run.outputs = outputs or {}
    run.extra = extra or {}
    return run


def test_count_tool_calls_in_run_no_messages() -> None:
    """Test counting tool calls when there are no messages."""
    run = create_mock_run()

    count = count_tool_calls_in_run(run)
    assert count == 0

    # Test counting tool calls when messages exist but no tool calls.
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
    ]
    run = create_mock_run(inputs={"messages": messages})

    count = count_tool_calls_in_run(run)
    assert count == 0

    # Test counting when `tool_calls` is empty list
    messages = [AIMessage(content="No tools", tool_calls=[])]
    run = create_mock_run(inputs={"messages": messages})

    count = count_tool_calls_in_run(run)
    assert count == 0


def test_count_tool_calls_in_run_with_tool_calls() -> None:
    """Test counting tool calls when they exist in messages."""
    tool_calls = [
        ToolCall(name="search", args={"query": "test"}, id="call_1"),
        ToolCall(name="calculator", args={"expression": "2+2"}, id="call_2"),
    ]

    messages = [
        HumanMessage(content="Search for test and calculate 2+2"),
        AIMessage(content="I'll help you with that", tool_calls=tool_calls),
    ]
    run = create_mock_run(inputs={"messages": messages})

    count = count_tool_calls_in_run(run)
    assert count == 2

    # Test counting tool calls when messages are in dict format.
    messages = [
        {"role": "human", "content": "Hello"},  # type: ignore[list-item]
        {  # type: ignore[list-item]
            "role": "assistant",
            "content": "Hi!",
            "tool_calls": [
                {"name": "search", "args": {"query": "test"}, "id": "call_1"},
            ],
        },
    ]
    run = create_mock_run(inputs={"messages": messages})

    count = count_tool_calls_in_run(run)
    assert count == 1


def test_count_tool_calls_in_run_outputs_too() -> None:
    """Test counting tool calls in both inputs and outputs."""
    input_tool_calls = [ToolCall(name="search", args={"query": "test"}, id="call_1")]
    output_tool_calls = [
        ToolCall(name="calculator", args={"expression": "2+2"}, id="call_2")
    ]

    input_messages = [AIMessage(content="Input", tool_calls=input_tool_calls)]
    output_messages = [AIMessage(content="Output", tool_calls=output_tool_calls)]

    run = create_mock_run(
        inputs={"messages": input_messages}, outputs={"messages": output_messages}
    )

    count = count_tool_calls_in_run(run)
    assert count == 2


def test_store_tool_call_count_in_run() -> None:
    """Test storing tool call count in run metadata."""
    tool_calls = [ToolCall(name="search", args={"query": "test"}, id="call_1")]
    messages = [AIMessage(content="Test", tool_calls=tool_calls)]
    run = create_mock_run(inputs={"messages": messages})

    count = store_tool_call_count_in_run(run)

    assert count == 1
    assert run.extra["tool_call_count"] == 1


def test_store_tool_call_count_always_store() -> None:
    """Test storing tool call count with `always_store=True`."""
    messages = [AIMessage(content="No tools")]
    run = create_mock_run(inputs={"messages": messages})

    count = store_tool_call_count_in_run(run, always_store=True)

    assert count == 0
    assert run.extra["tool_call_count"] == 0


def test_store_tool_call_count_no_tools_no_always_store() -> None:
    """Test that count is not stored when no tools and `always_store=False`."""
    messages = [AIMessage(content="No tools")]
    run = create_mock_run(inputs={"messages": messages})

    count = store_tool_call_count_in_run(run, always_store=False)

    assert count == 0
    assert "tool_call_count" not in run.extra


def test_store_tool_call_count_in_run_no_extra() -> None:
    """Test storing when `run.extra` is `None`."""
    tool_calls = [ToolCall(name="search", args={"query": "test"}, id="call_1")]
    messages = [AIMessage(content="Test", tool_calls=tool_calls)]
    run = create_mock_run(inputs={"messages": messages})
    run.extra = None

    count = store_tool_call_count_in_run(run)

    assert count == 1
    assert run.extra["tool_call_count"] == 1


def test_count_tool_calls_handles_none_inputs() -> None:
    """Test counting when inputs/outputs are `None`."""
    run = create_mock_run()
    run.inputs = None
    run.outputs = None

    count = count_tool_calls_in_run(run)
    assert count == 0


def test_count_tool_calls_mixed_message_types() -> None:
    """Test counting with mixed message object and `dict` types."""
    tool_calls_obj = [ToolCall(name="search", args={"query": "test"}, id="call_1")]

    messages = [
        AIMessage(content="Object message", tool_calls=tool_calls_obj),
        {
            "role": "assistant",
            "content": "Dict message",
            "tool_calls": [{"name": "calc", "args": {"expr": "1+1"}, "id": "call_2"}],
        },
    ]
    run = create_mock_run(inputs={"messages": messages})

    count = count_tool_calls_in_run(run)
    assert count == 2
