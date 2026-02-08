"""Tests for tool messages."""

from langchain_core.messages import ToolMessageChunk


def test_tool_message_chunk_merge_none_content() -> None:
    """Test that merging ToolMessageChunk with None content preserves None."""
    chunk1 = ToolMessageChunk(
        tool_call_id="call_123",
        content=None,
        artifact=None,
        additional_kwargs={},
        response_metadata={},
        id="msg1",
    )

    chunk2 = ToolMessageChunk(
        tool_call_id="call_123",
        content=None,
        artifact=None,
        additional_kwargs={},
        response_metadata={},
        id="msg1",
    )

    result = chunk1 + chunk2
    # Bug: result.content becomes "NoneNone" instead of None or ""
    assert result.content is None or result.content == "", (
        f"Expected None or '', got: {result.content!r}"
    )


def test_tool_message_chunk_merge_none_with_string() -> None:
    """Test merging ToolMessageChunk with None content and string content."""
    chunk1 = ToolMessageChunk(
        tool_call_id="call_123",
        content=None,
        artifact=None,
        additional_kwargs={},
        response_metadata={},
        id="msg1",
    )

    chunk2 = ToolMessageChunk(
        tool_call_id="call_123",
        content="hello",
        artifact=None,
        additional_kwargs={},
        response_metadata={},
        id="msg1",
    )

    result = chunk1 + chunk2
    assert result.content == "hello", f"Expected 'hello', got: {result.content!r}"


def test_tool_message_chunk_merge_string_with_none() -> None:
    """Test merging ToolMessageChunk with string content and None content."""
    chunk1 = ToolMessageChunk(
        tool_call_id="call_123",
        content="hello",
        artifact=None,
        additional_kwargs={},
        response_metadata={},
        id="msg1",
    )

    chunk2 = ToolMessageChunk(
        tool_call_id="call_123",
        content=None,
        artifact=None,
        additional_kwargs={},
        response_metadata={},
        id="msg1",
    )

    result = chunk1 + chunk2
    assert result.content == "hello", f"Expected 'hello', got: {result.content!r}"


def test_tool_message_chunk_merge_strings() -> None:
    """Test merging ToolMessageChunk with string contents."""
    chunk1 = ToolMessageChunk(
        tool_call_id="call_123",
        content="hello",
        artifact=None,
        additional_kwargs={},
        response_metadata={},
        id="msg1",
    )

    chunk2 = ToolMessageChunk(
        tool_call_id="call_123",
        content=" world",
        artifact=None,
        additional_kwargs={},
        response_metadata={},
        id="msg1",
    )

    result = chunk1 + chunk2
    expected = "hello world"
    assert result.content == expected, f"Expected {expected!r}, got: {result.content!r}"
