"""Test automatic tool call count storage in tracers."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.outputs import ChatGeneration, LLMResult
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

    run = MagicMock(spec=Run)
    run.id = "test-llm-run-id"
    run.run_type = "llm"
    run.extra = {}
    run.outputs = {}
    run.events = []
    run.end_time = None
    run.inputs = {}

    tracer.run_map[str(run.id)] = run

    tool_calls = [
        ToolCall(name="search", args={"query": "test"}, id="call_1"),
        ToolCall(name="calculator", args={"expression": "2+2"}, id="call_2"),
    ]
    message = AIMessage(content="Test", tool_calls=tool_calls)
    generation = ChatGeneration(message=message)
    response = LLMResult(generations=[[generation]])

    # Complete the LLM run (this should trigger automatic metadata storage)
    completed_run = tracer._complete_llm_run(response=response, run_id=run.id)

    assert "tool_call_count" in completed_run.extra
    assert completed_run.extra["tool_call_count"] == 2


def test_complete_llm_run_handles_no_tool_calls() -> None:
    """Test that `_complete_llm_run` handles runs with no tool calls gracefully."""
    tracer = MockTracerCore()

    run = MagicMock(spec=Run)
    run.id = "test-llm-run-id-no-tools"
    run.run_type = "llm"
    run.extra = {}
    run.outputs = {}
    run.events = []
    run.end_time = None
    run.inputs = {}

    tracer.run_map[str(run.id)] = run

    message = AIMessage(content="No tools here")
    generation = ChatGeneration(message=message)
    response = LLMResult(generations=[[generation]])

    completed_run = tracer._complete_llm_run(response=response, run_id=run.id)

    # Verify tool call count is not stored when there are no tool calls
    assert "tool_call_count" not in completed_run.extra


def test_complete_llm_run_handles_empty_generations() -> None:
    """Test that `_complete_llm_run` handles empty generations gracefully."""
    tracer = MockTracerCore()

    run = MagicMock(spec=Run)
    run.id = "test-llm-run-id-empty"
    run.run_type = "llm"
    run.extra = {}
    run.outputs = {}
    run.events = []
    run.end_time = None
    run.inputs = {}

    tracer.run_map[str(run.id)] = run

    response = LLMResult(generations=[[]])

    completed_run = tracer._complete_llm_run(response=response, run_id=run.id)

    assert "tool_call_count" not in completed_run.extra


def test_complete_llm_run_counts_tool_calls_from_multiple_generations() -> None:
    """Test that tool calls are counted from multiple generations."""
    tracer = MockTracerCore()

    run = MagicMock(spec=Run)
    run.id = "test-llm-run-id-multi"
    run.run_type = "llm"
    run.extra = {}
    run.outputs = {}
    run.events = []
    run.end_time = None
    run.inputs = {}

    tracer.run_map[str(run.id)] = run

    # Create multiple generations with tool calls
    tool_calls_1 = [ToolCall(name="search", args={"query": "test"}, id="call_1")]
    tool_calls_2 = [
        ToolCall(name="calculator", args={"expression": "2+2"}, id="call_2"),
        ToolCall(name="weather", args={"location": "NYC"}, id="call_3"),
    ]
    gen1 = ChatGeneration(message=AIMessage(content="Gen1", tool_calls=tool_calls_1))
    gen2 = ChatGeneration(message=AIMessage(content="Gen2", tool_calls=tool_calls_2))
    response = LLMResult(generations=[[gen1], [gen2]])

    completed_run = tracer._complete_llm_run(response=response, run_id=run.id)

    assert completed_run.extra["tool_call_count"] == 3


def test_complete_llm_run_handles_null_tool_calls() -> None:
    """Test that `_complete_llm_run` handles null `tool_calls` gracefully."""
    tracer = MockTracerCore()

    run = MagicMock(spec=Run)
    run.id = "test-llm-run-id-null-tools"
    run.run_type = "llm"
    run.extra = {}
    run.outputs = {}
    run.events = []
    run.end_time = None
    run.inputs = {}

    tracer.run_map[str(run.id)] = run

    message = AIMessage(content="Test with null tool_calls")
    generation = ChatGeneration(message=message)
    # Bypass Pydantic validation by directly setting attribute
    object.__setattr__(message, "tool_calls", None)
    response = LLMResult(generations=[[generation]])

    # Should not raise TypeError from len(None)
    completed_run = tracer._complete_llm_run(response=response, run_id=run.id)

    assert "tool_call_count" not in completed_run.extra
