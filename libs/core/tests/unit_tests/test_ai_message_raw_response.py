from typing import Any

import pytest

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    add_ai_message_chunks,
    message_chunk_to_message,
)


@pytest.fixture
def base_chunk() -> AIMessageChunk:
    """Create a base AIMessageChunk for reuse."""
    return AIMessageChunk(content="hello", raw_response={"delta": "hello"})


def test_ai_message_stores_raw_response() -> None:
    """Test that AIMessage correctly stores raw_response."""
    msg: AIMessage = AIMessage(content="hi", raw_response={"raw": "ok"})
    assert msg.raw_response == {"raw": "ok"}


def test_add_ai_message_chunks_merges_raw_response(base_chunk: AIMessageChunk) -> None:
    """Test merging of AIMessageChunk objects combines raw_response correctly."""
    chunk1: AIMessageChunk = base_chunk
    chunk2: AIMessageChunk = AIMessageChunk(
        content=" world",
        raw_response={"delta": " world"}
    )
    merged: AIMessageChunk = add_ai_message_chunks(chunk1, chunk2)
    assert merged.content == "hello world"
    assert isinstance(merged.raw_response, list)
    assert merged.raw_response == [{"delta": "hello"}, {"delta": " world"}]


def test_add_ai_message_chunks_handles_missing_raw_response() -> None:
    """Test merging when some chunks have missing raw_response."""
    c1: AIMessageChunk = AIMessageChunk(content="foo", raw_response={"delta": "foo"})
    c2: AIMessageChunk = AIMessageChunk(content="bar", raw_response=None)
    merged: AIMessageChunk = add_ai_message_chunks(c1, c2)
    # Single raw_response should be kept as dict, not list
    assert merged.raw_response == {"delta": "foo"}


def test_message_chunk_to_message_transfers_raw_response(
        base_chunk: AIMessageChunk) -> None:
    """Test that message_chunk_to_message preserves raw_response."""
    msg: AIMessage = message_chunk_to_message(base_chunk)
    assert isinstance(msg, AIMessage)
    # raw_response should be preserved
    assert msg.raw_response == {"delta": "hello"}


def test_message_chunk_to_message_ignores_non_chunk_input() -> None:
    """Test that message_chunk_to_message passes through non-chunk inputs."""
    raw: AIMessage = AIMessage(content="hi", raw_response={"data": 1})
    result: AIMessage = message_chunk_to_message(raw)
    # Should simply pass through
    assert result is raw


def test_empty_raw_response_not_present_in_serialized_message() -> None:
    """Test that raw_response is omitted when serializing message with None."""
    msg: AIMessage = AIMessage(content="test", raw_response=None)
    attrs: dict[str, Any] = msg.lc_attributes
    assert "raw_response" not in attrs


def test_invalid_input_handling_for_merging_different_types() -> None:
    """Test add_ai_message_chunks handles single or empty merge cases gracefully."""
    chunk: AIMessageChunk = AIMessageChunk(content="hello")
    # Should be unaffected even when others list is empty
    merged: AIMessageChunk = add_ai_message_chunks(chunk)
    assert merged.content == "hello"
