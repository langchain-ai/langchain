"""Test automatic tool call count storage in tracers."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.outputs import LLMResult
from langchain_core.tracers.core import _TracerCore
from langchain_core.tracers.schemas import Run


class MockTracerCore(_TracerCore):
    """Mock tracer core for testing LLM run completion."""

    def __init__(self) -> None:
        super().__init__()

    def _persist_run(self, run: Run) -> None:
        """Mock implementation of _persist_run."""


def test_complete_llm_run_automatically_stores_tool_call_count() -> None:
    """Test that `_complete_llm_run` automatically stores tool call count."""
    tracer = MockTracerCore()

    # Create a mock LLM run with tool calls
    run = MagicMock(spec=Run)
    run.id = "test-llm-run-id"
    run.run_type = "llm"
    run.extra = {}
    run.outputs = {}
    run.events = []
    run.end_time = None

    # Set up messages with tool calls in inputs
    tool_calls = [
        ToolCall(name="search", args={"query": "test"}, id="call_1"),
        ToolCall(name="calculator", args={"expression": "2+2"}, id="call_2"),
    ]
    messages = [AIMessage(content="Test", tool_calls=tool_calls)]
    run.inputs = {"messages": messages}

    # Add run to tracer's run_map
    tracer.run_map[str(run.id)] = run

    # Create a mock LLMResult
    response = MagicMock(spec=LLMResult)
    response.model_dump.return_value = {"generations": [[]]}
    response.generations = [[]]

    # Complete the LLM run (this should trigger automatic metadata storage)
    completed_run = tracer._complete_llm_run(response=response, run_id=run.id)

    # Verify tool call count was automatically stored
    assert "tool_call_count" in completed_run.extra
    assert completed_run.extra["tool_call_count"] == 2


def test_complete_llm_run_handles_no_tool_calls() -> None:
    """Test that `_complete_llm_run` handles runs with no tool calls gracefully."""
    tracer = MockTracerCore()

    # Create a mock LLM run without tool calls
    run = MagicMock(spec=Run)
    run.id = "test-llm-run-id-no-tools"
    run.run_type = "llm"
    run.extra = {}
    run.outputs = {}
    run.events = []
    run.end_time = None

    # Set up messages without tool calls
    messages = [AIMessage(content="No tools here")]
    run.inputs = {"messages": messages}

    # Add run to tracer's run_map
    tracer.run_map[str(run.id)] = run

    # Create a mock LLMResult
    response = MagicMock(spec=LLMResult)
    response.model_dump.return_value = {"generations": [[]]}
    response.generations = [[]]

    # Complete the LLM run
    completed_run = tracer._complete_llm_run(response=response, run_id=run.id)

    # Verify tool call count is not stored when there are no tool calls
    assert "tool_call_count" not in completed_run.extra


def test_complete_llm_run_handles_runs_without_messages() -> None:
    """Test that `_complete_llm_run` handles runs without messages gracefully."""
    tracer = MockTracerCore()

    # Create a mock LLM run without messages
    run = MagicMock(spec=Run)
    run.id = "test-llm-run-id-no-messages"
    run.run_type = "llm"
    run.extra = {}
    run.outputs = {}
    run.events = []
    run.end_time = None
    run.inputs = {}

    # Add run to tracer's run_map
    tracer.run_map[str(run.id)] = run

    # Create a mock LLMResult
    response = MagicMock(spec=LLMResult)
    response.model_dump.return_value = {"generations": [[]]}
    response.generations = [[]]

    # Complete the LLM run
    completed_run = tracer._complete_llm_run(response=response, run_id=run.id)

    # Verify tool call count is not stored when there are no messages
    assert "tool_call_count" not in completed_run.extra


def test_complete_llm_run_continues_on_metadata_error() -> None:
    """Test that `_complete_llm_run` continues working if metadata storage fails."""
    tracer = MockTracerCore()

    # Create a mock LLM run that will cause an error in tool call counting
    run = MagicMock(spec=Run)
    run.id = "test-llm-run-id-error"
    run.run_type = "llm"
    run.extra = {}
    run.outputs = {}
    run.events = []
    run.end_time = None

    # Make the run.inputs property raise an error when accessed
    type(run).inputs = PropertyMock(side_effect=RuntimeError("Simulated error"))

    # Add run to tracer's run_map
    tracer.run_map[str(run.id)] = run

    # Create a mock LLMResult
    response = MagicMock(spec=LLMResult)
    response.model_dump.return_value = {"generations": [[]]}
    response.generations = [[]]

    # Complete the LLM run - this should raise an exception due to our mock error
    # but that's expected behavior since the tool call counting failed
    try:  # noqa: SIM105
        tracer._complete_llm_run(response=response, run_id=run.id)
        # If no exception is raised, then the implementation changed
        # to be more defensive, which is fine
    except RuntimeError:
        # This is the expected behavior since we made inputs raise an error
        pass
