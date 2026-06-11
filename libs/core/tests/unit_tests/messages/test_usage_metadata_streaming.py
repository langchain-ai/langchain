"""Regression test for https://github.com/langchain-ai/langchain/issues/31351."""

from langchain_core.messages import AIMessageChunk
from langchain_core.messages.ai import add_ai_message_chunks


def test_usage_metadata_summed_for_independent_merges() -> None:
    """Independent chunks with different input_tokens should be summed."""
    left = AIMessageChunk(
        content="",
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )
    right = AIMessageChunk(
        content="",
        usage_metadata={"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
    )

    merged = add_ai_message_chunks(left, right)

    assert merged.usage_metadata is not None
    assert merged.usage_metadata["input_tokens"] == 5
    assert merged.usage_metadata["output_tokens"] == 7
    assert merged.usage_metadata["total_tokens"] == 12


def test_usage_metadata_streaming_not_summed() -> None:
    """Cumulative streaming chunks (same input, increasing total) use last."""
    chunk1 = AIMessageChunk(
        content="Hello",
        usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
    )
    chunk2 = AIMessageChunk(
        content=" world",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
    chunk3 = AIMessageChunk(
        content="!",
        usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    )

    merged = add_ai_message_chunks(chunk1, chunk2, chunk3)

    assert merged.usage_metadata is not None
    assert merged.usage_metadata["input_tokens"] == 10
    assert merged.usage_metadata["output_tokens"] == 3
    assert merged.usage_metadata["total_tokens"] == 13


def test_usage_metadata_none_chunks() -> None:
    """If no chunks carry usage_metadata, the merged result should be None."""
    chunk1 = AIMessageChunk(content="Hello")
    chunk2 = AIMessageChunk(content=" world")

    merged = add_ai_message_chunks(chunk1, chunk2)

    assert merged.usage_metadata is None


def test_usage_metadata_partial_chunks() -> None:
    """Only some chunks have usage_metadata — use the single non-null one."""
    chunk1 = AIMessageChunk(content="Hello")
    chunk2 = AIMessageChunk(
        content=" world",
        usage_metadata={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
    )
    chunk3 = AIMessageChunk(content="!")

    merged = add_ai_message_chunks(chunk1, chunk2, chunk3)

    assert merged.usage_metadata is not None
    assert merged.usage_metadata["input_tokens"] == 5
    assert merged.usage_metadata["output_tokens"] == 10
    assert merged.usage_metadata["total_tokens"] == 15


def test_usage_metadata_last_chunk_only() -> None:
    """Common pattern: only the final streaming chunk carries usage."""
    chunk1 = AIMessageChunk(content="Hello")
    chunk2 = AIMessageChunk(content=" world")
    chunk3 = AIMessageChunk(
        content="",
        usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    )

    merged = add_ai_message_chunks(chunk1, chunk2, chunk3)

    assert merged.usage_metadata is not None
    assert merged.usage_metadata["input_tokens"] == 10
    assert merged.usage_metadata["output_tokens"] == 20
    assert merged.usage_metadata["total_tokens"] == 30
