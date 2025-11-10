"""Test groq block translator."""

from typing import cast

import pytest

from langchain_core.messages import AIMessage
from langchain_core.messages import content as types
from langchain_core.messages.base import _extract_reasoning_from_additional_kwargs
from langchain_core.messages.block_translators import PROVIDER_TRANSLATORS
from langchain_core.messages.block_translators.groq import (
    _parse_code_json,
    translate_content,
)


def test_groq_translator_imports() -> None:
    """Test that groq translator module can be imported successfully.

    This test ensures that all required dependencies for the groq translator
    are available, including the _extract_reasoning_from_additional_kwargs function.
    """
    # If imports at top of file succeeded, the translator dependencies are available
    assert callable(translate_content)
    assert callable(_extract_reasoning_from_additional_kwargs)


def test_groq_translator_registered() -> None:
    """Test that groq translator is properly registered."""
    assert "groq" in PROVIDER_TRANSLATORS
    assert "translate_content" in PROVIDER_TRANSLATORS["groq"]
    assert "translate_content_chunk" in PROVIDER_TRANSLATORS["groq"]


def test_extract_reasoning_from_additional_kwargs_exists() -> None:
    """Test that _extract_reasoning_from_additional_kwargs can be imported."""
    # Verify it's callable
    assert callable(_extract_reasoning_from_additional_kwargs)


def test_groq_translate_content_basic() -> None:
    """Test basic groq content translation."""
    # Test with simple text message
    message = AIMessage(content="Hello world")
    blocks = translate_content(message)

    assert isinstance(blocks, list)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "Hello world"


def test_groq_translate_content_with_reasoning() -> None:
    """Test groq content translation with reasoning content."""
    # Test with reasoning content in additional_kwargs
    message = AIMessage(
        content="Final answer",
        additional_kwargs={"reasoning_content": "Let me think about this..."},
    )
    blocks = translate_content(message)

    assert isinstance(blocks, list)
    assert len(blocks) == 2

    # First block should be reasoning
    assert blocks[0]["type"] == "reasoning"
    assert blocks[0]["reasoning"] == "Let me think about this..."

    # Second block should be text
    assert blocks[1]["type"] == "text"
    assert blocks[1]["text"] == "Final answer"


def test_groq_translate_content_with_tool_calls() -> None:
    """Test groq content translation with tool calls."""
    # Test with tool calls
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search",
                "args": {"query": "test"},
                "id": "call_123",
            }
        ],
    )
    blocks = translate_content(message)

    assert isinstance(blocks, list)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "tool_call"
    assert blocks[0]["name"] == "search"
    assert blocks[0]["args"] == {"query": "test"}
    assert blocks[0]["id"] == "call_123"


def test_groq_translate_content_with_executed_tools() -> None:
    """Test groq content translation with executed tools (built-in tools)."""
    # Test with executed_tools in additional_kwargs (Groq built-in tools)
    message = AIMessage(
        content="",
        additional_kwargs={
            "executed_tools": [
                {
                    "type": "python",
                    "arguments": '{"code": "print(\\"hello\\")"}',
                    "output": "hello\\n",
                }
            ]
        },
    )
    blocks = translate_content(message)

    assert isinstance(blocks, list)
    # Should have server_tool_call and server_tool_result
    assert len(blocks) >= 2

    # Check for server_tool_call
    tool_call_blocks = [
        cast("types.ServerToolCall", b)
        for b in blocks
        if b.get("type") == "server_tool_call"
    ]
    assert len(tool_call_blocks) == 1
    assert tool_call_blocks[0]["name"] == "code_interpreter"
    assert "code" in tool_call_blocks[0]["args"]

    # Check for server_tool_result
    tool_result_blocks = [
        cast("types.ServerToolResult", b)
        for b in blocks
        if b.get("type") == "server_tool_result"
    ]
    assert len(tool_result_blocks) == 1
    assert tool_result_blocks[0]["output"] == "hello\\n"
    assert tool_result_blocks[0]["status"] == "success"


def test_parse_code_json() -> None:
    """Test the _parse_code_json helper function."""
    # Test valid code JSON
    result = _parse_code_json('{"code": "print(\'hello\')"}')
    assert result == {"code": "print('hello')"}

    # Test code with unescaped quotes (Groq format)
    result = _parse_code_json('{"code": "print("hello")"}')
    assert result == {"code": 'print("hello")'}

    # Test invalid format raises ValueError
    with pytest.raises(ValueError, match="Could not extract Python code"):
        _parse_code_json('{"invalid": "format"}')
