"""Utility functions for working with Run objects and tracers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tracers.schemas import Run


def count_tool_calls_in_run(run: Run) -> int:
    """Count tool calls in a `Run` object by examining messages.

    Args:
        run: The `Run` object to examine.

    Returns:
        The total number of tool calls found in the run's messages.
    """
    tool_call_count = 0

    def _count_tool_calls_in_messages(messages: list) -> int:
        count = 0
        for msg in messages:
            if hasattr(msg, "tool_calls"):
                tool_calls = getattr(msg, "tool_calls", [])
                count += len(tool_calls)
            elif isinstance(msg, dict) and "tool_calls" in msg:
                tool_calls = msg.get("tool_calls", [])
                count += len(tool_calls)
        return count

    # Check inputs for messages containing tool calls
    inputs = getattr(run, "inputs", {})
    if isinstance(inputs, dict) and "messages" in inputs:
        messages = inputs["messages"]
        if messages:
            tool_call_count += _count_tool_calls_in_messages(messages)

    outputs = getattr(run, "outputs", {})
    if isinstance(outputs, dict) and "messages" in outputs:
        messages = outputs["messages"]
        if messages:
            tool_call_count += _count_tool_calls_in_messages(messages)

    return tool_call_count


def store_tool_call_count_in_run(run: Run, *, always_store: bool = False) -> int:
    """Count tool calls in a `Run` and store the count in run metadata.

    Args:
        run: The `Run` object to analyze and modify.
        always_store: If `True`, always store the count even if `0`. If `False`, only
            store when there are tool calls.

    Returns:
        The number of tool calls found and stored.
    """
    tool_call_count = count_tool_calls_in_run(run)

    # Only store if there are tool calls or if explicitly requested
    if tool_call_count > 0 or always_store:
        if run.extra is None:
            run.extra = {}
        run.extra["tool_call_count"] = tool_call_count

    return tool_call_count
