#!/usr/bin/env python3
"""
Unit tests for the RunnableSequence.bind_tools method
Tests the fix for issue #28848
"""

import pytest
from pydantic import BaseModel, Field
from typing import Any, Callable, Union

# Import from our local libs
import sys
sys.path.insert(0, '/home/nav/langchain/libs/core')

from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tools import StructuredTool


class MockResponseModel(BaseModel):
    value: str = Field(description="A test value")


def mock_tool_function(x: int) -> bool:
    """A mock tool function for testing."""
    return True


def test_sequence_bind_tools_with_bindable_first_step():
    """Test that bind_tools works when the first step has bind_tools method."""
    
    # Create a mock first step that supports bind_tools
    def mock_model_func(x):
        return x
        
    mock_model = RunnableLambda(mock_model_func)
    
    # Mock the bind_tools method
    def bind_tools_method(tools, **kwargs):
        return RunnableLambda(lambda x: f"bound_with_{len(tools)}_tools({x})")
        
    mock_model.bind_tools = bind_tools_method
    
    # Create parser
    mock_parser = RunnableLambda(lambda x: f"parsed({x})")
    
    # Create sequence (like with_structured_output does)
    sequence = RunnableSequence(mock_model, mock_parser)
    
    # Create tool
    tool = StructuredTool.from_function(
        func=mock_tool_function,
        name="mock_tool",
        description="A mock tool",
    )
    
    # Test bind_tools
    result_sequence = sequence.bind_tools([tool])
    
    # Should return a new RunnableSequence
    assert isinstance(result_sequence, RunnableSequence)
    assert len(result_sequence.steps) == 2
    
    # The last step should be the same parser
    assert result_sequence.last == mock_parser


def test_sequence_bind_tools_without_bindable_first_step():
    """Test that bind_tools raises proper error when first step doesn't support it."""
    
    # Create sequence with steps that don't support bind_tools
    step1 = RunnableLambda(lambda x: x)
    step2 = RunnableLambda(lambda x: x)
    
    sequence = RunnableSequence(step1, step2)
    
    tool = StructuredTool.from_function(
        func=mock_tool_function,
        name="mock_tool",
        description="A mock tool",
    )
    
    # Should raise AttributeError with helpful message
    with pytest.raises(AttributeError) as exc_info:
        sequence.bind_tools([tool])
    
    assert "bind_tools" in str(exc_info.value)
    assert "RunnableSequence" in str(exc_info.value)


def test_sequence_bind_tools_with_middle_steps():
    """Test bind_tools with a sequence that has middle steps."""
    
    # Create a mock first step that supports bind_tools
    mock_model = RunnableLambda(lambda x: x)
    
    def bind_tools_method(tools, **kwargs):
        return RunnableLambda(lambda x: f"bound({x})")
        
    mock_model.bind_tools = bind_tools_method
    
    # Create middle and last steps
    middle_step = RunnableLambda(lambda x: f"middle({x})")
    last_step = RunnableLambda(lambda x: f"last({x})")
    
    # Create sequence with middle steps
    sequence = RunnableSequence(mock_model, middle_step, last_step)
    
    tool = StructuredTool.from_function(
        func=mock_tool_function,
        name="mock_tool", 
        description="A mock tool",
    )
    
    # Test bind_tools
    result_sequence = sequence.bind_tools([tool])
    
    assert isinstance(result_sequence, RunnableSequence)
    assert len(result_sequence.steps) == 3
    assert result_sequence.middle == [middle_step]
    assert result_sequence.last == last_step


def test_sequence_bind_tools_preserves_name():
    """Test that bind_tools preserves the sequence name."""
    
    mock_model = RunnableLambda(lambda x: x)
    mock_model.bind_tools = lambda tools, **kwargs: RunnableLambda(lambda x: x)
    
    mock_parser = RunnableLambda(lambda x: x)
    
    # Create sequence with a name
    sequence = RunnableSequence(mock_model, mock_parser, name="test_sequence")
    
    tool = StructuredTool.from_function(
        func=mock_tool_function,
        name="mock_tool",
        description="A mock tool",
    )
    
    result_sequence = sequence.bind_tools([tool])
    
    assert result_sequence.name == "test_sequence"


if __name__ == "__main__":
    # Run the tests manually if executed directly
    test_sequence_bind_tools_with_bindable_first_step()
    print("✅ test_sequence_bind_tools_with_bindable_first_step passed")
    
    test_sequence_bind_tools_without_bindable_first_step()
    print("✅ test_sequence_bind_tools_without_bindable_first_step passed")
    
    test_sequence_bind_tools_with_middle_steps()
    print("✅ test_sequence_bind_tools_with_middle_steps passed")
    
    test_sequence_bind_tools_preserves_name()
    print("✅ test_sequence_bind_tools_preserves_name passed")
    
    print("\n🎉 All tests passed!")