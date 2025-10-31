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

    # Check inputs for messages containing tool calls
    inputs = getattr(run, "inputs", {}) or {}
    if isinstance(inputs, dict) and "messages" in inputs:
        messages = inputs["messages"]
        if messages:
            for msg in messages:
                # Handle both dict and object representations
                if hasattr(msg, "tool_calls"):
                    tool_calls = getattr(msg, "tool_calls", [])
                    if tool_calls:
                        tool_call_count += len(tool_calls)
                elif isinstance(msg, dict) and "tool_calls" in msg:
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        tool_call_count += len(tool_calls)

    # Also check outputs for completeness
    outputs = getattr(run, "outputs", {}) or {}
    if isinstance(outputs, dict) and "messages" in outputs:
        messages = outputs["messages"]
        if messages:
            for msg in messages:
                if hasattr(msg, "tool_calls"):
                    tool_calls = getattr(msg, "tool_calls", [])
                    if tool_calls:
                        tool_call_count += len(tool_calls)
                elif isinstance(msg, dict) and "tool_calls" in msg:
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        tool_call_count += len(tool_calls)

    return tool_call_count


def store_tool_call_count_in_run(run: Run, *, always_store: bool = False) -> int:
    """Count tool calls in a `Run` and store the count in run metadata.

    Args:
        run: The `Run` object to analyze and modify.
        always_store: If `True`, always store the count even if `0`.
            If `False`, only store when there are tool calls.

    Returns:
        The number of tool calls found and stored.
    """
    tool_call_count = count_tool_calls_in_run(run)

    # Only store if there are tool calls or if explicitly requested
    if tool_call_count > 0 or always_store:
        # Store in run.extra for easy access
        if not hasattr(run, "extra") or run.extra is None:
            run.extra = {}
        run.extra["tool_call_count"] = tool_call_count

    return tool_call_count


def get_tool_call_count_from_run(run: Run) -> int | None:
    """Get the tool call count from run metadata if available.

    Args:
        run: The `Run` object to check.

    Returns:
        The tool call count if stored in metadata, otherwise `None`.
    """
    extra = getattr(run, "extra", {}) or {}
    if isinstance(extra, dict):
        return extra.get("tool_call_count")
    return None
