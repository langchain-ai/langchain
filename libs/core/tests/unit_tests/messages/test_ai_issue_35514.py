"""Test for issue #35514: Tool call with empty args from SSE fragmentation."""

from langchain_core.messages import AIMessageChunk
from langchain_core.messages.tool import invalid_tool_call as create_invalid_tool_call
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk


def test_tool_call_chunks_with_fragmented_args() -> None:
    """Test that tool call chunks with fragmented arguments are handled correctly.

    When streaming tool calls via SSE, arguments can be fragmented across
    multiple chunks. The first chunk may have an empty args string, which
    should NOT be parsed as a valid tool call with empty dict {}.
    Instead, it should be treated as invalid until all chunks are accumulated.

    This test reproduces the issue described in #35514.
    """
    # Simulate SSE fragmentation: first chunk has tool name but empty args
    chunk1 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            create_tool_call_chunk(name="my_tool", args="", id="call_123", index=0)
        ],
    )

    # First chunk with empty args should NOT create a valid tool call
    assert chunk1.tool_calls == []
    assert chunk1.invalid_tool_calls == [
        create_invalid_tool_call(name="my_tool", args="", id="call_123", error=None)
    ]

    # Subsequent chunks contain the actual arguments
    chunk2 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            create_tool_call_chunk(name=None, args='{"url": "', id=None, index=0)
        ],
    )

    chunk3 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            create_tool_call_chunk(
                name=None, args='http://example.com"}', id=None, index=0
            )
        ],
    )

    # Accumulate all chunks
    accumulated = chunk1 + chunk2 + chunk3

    # After accumulation, we should have a valid tool call with complete arguments
    assert accumulated.tool_calls == [
        create_tool_call(
            name="my_tool", args={"url": "http://example.com"}, id="call_123"
        )
    ]
    assert accumulated.invalid_tool_calls == []


def test_tool_call_chunks_with_none_args() -> None:
    """Test that tool call chunks with None args are handled correctly.

    When args is None (not empty string), it should be treated as empty dict.
    This is different from empty string which should fail parsing.
    """
    # Chunk with None args should create a valid tool call with empty dict
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            create_tool_call_chunk(name="my_tool", args=None, id="call_123", index=0)
        ],
    )

    assert chunk.tool_calls == [
        create_tool_call(name="my_tool", args={}, id="call_123")
    ]
    assert chunk.invalid_tool_calls == []


def test_tool_call_chunks_partial_json() -> None:
    """Test that partial JSON in tool call chunks is handled correctly."""
    # Partial JSON that can be completed will create a valid tool call with empty dict
    # This is expected behavior of parse_partial_json
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            create_tool_call_chunk(
                name="my_tool", args='{"incomplete": ', id="call_123", index=0
            )
        ],
    )

    # parse_partial_json completes the partial JSON to {}
    assert chunk.tool_calls == [
        create_tool_call(name="my_tool", args={}, id="call_123")
    ]
    assert chunk.invalid_tool_calls == []
