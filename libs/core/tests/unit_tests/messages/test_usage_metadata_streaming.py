"""Regression test for https://github.com/langchain-ai/langchain/issues/31351."""

from langchain_core.messages import AIMessageChunk
from langchain_core.messages.ai import add_ai_message_chunks


def test_usage_metadata_streaming_not_summed() -> None:
    """Merged usage_metadata should equal the last chunk, not the sum."""
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
    """If only some chunks carry usage_metadata, use the last non-null one."""
    chunk1 = AIMessageChunk(content="Hello")

    chunk2 = AIMessageChunk(
        content=" world",
        usage_metadata={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
    )

    chunk3 = AIMessageChunk(content="!")

    merged = add_ai_message_chunks(chunk1, chunk2, chunk3)

    # Last non-null is chunk2
    assert merged.usage_metadata is not None
    assert merged.usage_metadata["input_tokens"] == 5
    assert merged.usage_metadata["output_tokens"] == 10
    assert merged.usage_metadata["total_tokens"] == 15
