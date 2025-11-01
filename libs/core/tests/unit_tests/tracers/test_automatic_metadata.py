"""Test automatic tool call count storage in tracers."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run


class MockTracer(BaseTracer):
    """Mock tracer for testing automatic metadata storage."""

    def __init__(self) -> None:
        super().__init__()
        self.persisted_runs: list[Run] = []

    def _persist_run(self, run: Run) -> None:
        """Store the run for inspection."""
        self.persisted_runs.append(run)


def test_base_tracer_automatically_stores_tool_call_count() -> None:
    """Test that `BaseTracer` automatically stores tool call count."""
    tracer = MockTracer()

    # Create a mock run with tool calls
    run = MagicMock(spec=Run)
    run.id = "test-run-id"
    run.parent_run_id = None  # Root run, will be persisted
    run.extra = {}

    # Set up messages with tool calls
    tool_calls = [
        ToolCall(name="search", args={"query": "test"}, id="call_1"),
        ToolCall(name="calculator", args={"expression": "2+2"}, id="call_2"),
    ]
    messages = [AIMessage(content="Test", tool_calls=tool_calls)]
    run.inputs = {"messages": messages}
    run.outputs = {}

    # Add run to tracer's run_map to simulate it being tracked
    tracer.run_map[str(run.id)] = run

    # End the trace (this should trigger automatic metadata storage)
    tracer._end_trace(run)

    # Verify tool call count was automatically stored
    assert "tool_call_count" in run.extra
    assert run.extra["tool_call_count"] == 2

    # Verify the run was persisted
    assert len(tracer.persisted_runs) == 1
    assert tracer.persisted_runs[0] == run


def test_base_tracer_handles_no_tool_calls() -> None:
    """Test that `BaseTracer` handles runs with no tool calls gracefully."""
    tracer = MockTracer()

    # Create a mock run without tool calls
    run = MagicMock(spec=Run)
    run.id = "test-run-id-no-tools"
    run.parent_run_id = None
    run.extra = {}

    # Set up messages without tool calls
    messages = [AIMessage(content="No tools here")]
    run.inputs = {"messages": messages}
    run.outputs = {}

    # Add run to tracer's run_map
    tracer.run_map[str(run.id)] = run

    # End the trace
    tracer._end_trace(run)

    # Verify tool call count is not stored when there are no tool calls
    assert "tool_call_count" not in run.extra


def test_base_tracer_handles_runs_without_messages() -> None:
    """Test that `BaseTracer` handles runs without messages gracefully."""
    tracer = MockTracer()

    # Create a mock run without messages
    run = MagicMock(spec=Run)
    run.id = "test-run-id-no-messages"
    run.parent_run_id = None
    run.extra = {}
    run.inputs = {}
    run.outputs = {}

    # Add run to tracer's run_map
    tracer.run_map[str(run.id)] = run

    # End the trace
    tracer._end_trace(run)

    # Verify tool call count is not stored when there are no messages
    assert "tool_call_count" not in run.extra


def test_base_tracer_doesnt_break_on_metadata_error() -> None:
    """Test that `BaseTracer` continues working if metadata storage fails."""
    tracer = MockTracer()

    # Create a mock run that will cause an error in tool call counting
    run = MagicMock(spec=Run)
    run.id = "test-run-id-error"
    run.parent_run_id = None
    run.extra = {}

    # Make the run.inputs property raise an error when accessed
    type(run).inputs = PropertyMock(side_effect=RuntimeError("Simulated error"))

    # Add run to tracer's run_map
    tracer.run_map[str(run.id)] = run

    # End the trace - this should not raise an exception
    tracer._end_trace(run)

    # The run should still be persisted despite the metadata error
    assert len(tracer.persisted_runs) == 1
    assert tracer.persisted_runs[0] == run
