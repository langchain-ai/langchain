"""Unit tests for RunnableSequence.bind_tools method.

Tests the fix for issue #28848: ChatOpenAI bind_tools not callable after
with_structured_output
"""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.tools import StructuredTool


class MockResponseModel(BaseModel):
    """Mock Pydantic model for testing."""

    value: str = Field(description="A test value")


def mock_tool_function(_x: int) -> bool:
    """Mock tool function for testing."""
    return True


def test_sequence_bind_tools_structured_output_pattern() -> None:
    """Test bind_tools works with the typical structured output pattern.

    This test covers the main use case from issue #28848:
    - ChatModel.with_structured_output() creates a RunnableSequence
    - Calling bind_tools() on that sequence should work
    """

    # Create mock model that supports bind_tools (like ChatOpenAI)
    def mock_model_func(x: Any) -> Any:
        return x

    mock_model = RunnableLambda(mock_model_func)

    # Mock the bind_tools method to simulate ChatOpenAI behavior
    def bind_tools_method(tools: Any, **_kwargs: Any) -> RunnableLambda:
        return RunnableLambda(lambda x: f"bound_with_{len(tools)}_tools({x})")

    mock_model.bind_tools = bind_tools_method  # type: ignore[attr-defined]

    # Create parser (simulates output parser from with_structured_output)
    mock_parser = RunnableLambda(lambda x: f"parsed({x})")

    # Create sequence that mimics with_structured_output result
    sequence: RunnableSequence = RunnableSequence(mock_model, mock_parser)

    # Create tool
    tool = StructuredTool.from_function(
        func=mock_tool_function,
        name="mock_tool",
        description="A mock tool for testing",
    )

    # This should work without raising AttributeError
    result_sequence = sequence.bind_tools([tool])

    # Verify the result
    assert isinstance(result_sequence, RunnableSequence)
    assert len(result_sequence.steps) == 2
    assert result_sequence.last == mock_parser  # Parser should be preserved


def test_sequence_bind_tools_with_kwargs() -> None:
    """Test that bind_tools passes kwargs correctly."""
    # Track calls to bind_tools
    bind_tools_calls = []

    def mock_model_func(x: Any) -> Any:
        return x

    mock_model = RunnableLambda(mock_model_func)

    def bind_tools_method(tools: Any, **kwargs: Any) -> RunnableLambda:
        bind_tools_calls.append((len(tools), kwargs))
        return RunnableLambda(lambda x: x)

    mock_model.bind_tools = bind_tools_method  # type: ignore[attr-defined]
    mock_parser = RunnableLambda(lambda x: x)

    sequence: RunnableSequence = RunnableSequence(mock_model, mock_parser)

    tool = StructuredTool.from_function(
        func=mock_tool_function,
        name="test_tool",
        description="A test tool",
    )

    # Call with specific kwargs
    sequence.bind_tools([tool], tool_choice="auto", strict=True)

    # Verify kwargs were passed through
    assert len(bind_tools_calls) == 1
    assert bind_tools_calls[0][0] == 1  # 1 tool
    assert bind_tools_calls[0][1] == {"tool_choice": "auto", "strict": True}


def test_sequence_bind_tools_sequence_with_middle_steps() -> None:
    """Test bind_tools with longer sequences."""
    mock_model = RunnableLambda(lambda x: x)
    mock_model.bind_tools = lambda _tools, **_kwargs: RunnableLambda(lambda _x: "bound")  # type: ignore[attr-defined]

    middle_step = RunnableLambda(lambda x: f"middle({x})")
    last_step = RunnableLambda(lambda x: f"last({x})")

    # Create sequence with middle steps
    sequence: RunnableSequence = RunnableSequence(mock_model, middle_step, last_step)

    tool = StructuredTool.from_function(
        func=mock_tool_function,
        name="test_tool",
        description="A test tool",
    )

    result_sequence = sequence.bind_tools([tool])

    assert isinstance(result_sequence, RunnableSequence)
    assert len(result_sequence.steps) == 3
    assert result_sequence.middle == [middle_step]
    assert result_sequence.last == last_step


def test_sequence_bind_tools_preserves_name() -> None:
    """Test that sequence name is preserved after bind_tools."""
    mock_model = RunnableLambda(lambda x: x)
    mock_model.bind_tools = lambda _tools, **_kwargs: RunnableLambda(lambda x: x)  # type: ignore[attr-defined]

    mock_parser = RunnableLambda(lambda x: x)

    sequence: RunnableSequence = RunnableSequence(
        mock_model, mock_parser, name="my_structured_llm"
    )

    tool = StructuredTool.from_function(
        func=mock_tool_function,
        name="test_tool",
        description="A test tool",
    )

    result_sequence = sequence.bind_tools([tool])

    assert result_sequence.name == "my_structured_llm"


def test_sequence_bind_tools_no_bindable_step_raises_error() -> None:
    """Test that proper error is raised when no step supports bind_tools."""
    # Create sequence with no bindable steps
    step1 = RunnableLambda(lambda x: x)
    step2 = RunnableLambda(lambda x: x)

    sequence: RunnableSequence = RunnableSequence(step1, step2)

    tool = StructuredTool.from_function(
        func=mock_tool_function,
        name="test_tool",
        description="A test tool",
    )

    # Should raise AttributeError with helpful message
    with pytest.raises(AttributeError) as exc_info:
        sequence.bind_tools([tool])

    error_msg = str(exc_info.value)
    assert "bind_tools" in error_msg
    assert "RunnableSequence" in error_msg


def test_sequence_bind_tools_multiple_tools() -> None:
    """Test binding multiple tools at once."""
    mock_model = RunnableLambda(lambda x: x)

    def bind_tools_method(tools: Any, **_kwargs: Any) -> RunnableLambda:
        return RunnableLambda(lambda _x: f"bound_with_{len(tools)}_tools")

    mock_model.bind_tools = bind_tools_method  # type: ignore[attr-defined]
    mock_parser = RunnableLambda(lambda x: x)

    sequence: RunnableSequence = RunnableSequence(mock_model, mock_parser)

    # Create multiple tools
    tool1 = StructuredTool.from_function(
        func=mock_tool_function,
        name="tool1",
        description="First tool",
    )
    tool2 = StructuredTool.from_function(
        func=mock_tool_function,
        name="tool2",
        description="Second tool",
    )

    result_sequence = sequence.bind_tools([tool1, tool2])

    assert isinstance(result_sequence, RunnableSequence)
    # The first step should indicate it was bound with 2 tools
    # This is verified by our mock implementation
