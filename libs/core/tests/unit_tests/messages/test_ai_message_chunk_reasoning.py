"""Unit tests for reasoning_content field in AIMessageChunk.

These tests verify that reasoning_content is a first-class field that:
1. Can be set directly on AIMessageChunk
2. Is properly merged when chunks are combined
3. Is accessible via public API (not hidden in additional_kwargs)
4. Works correctly with other fields (content, tool_calls, etc.)

This addresses the architectural issue where reasoning was previously hidden
in additional_kwargs, making it inconsistent and not part of the public API.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.messages.ai import add_ai_message_chunks


class TestReasoningContentField:
    """Test that reasoning_content is a first-class field on AIMessageChunk."""

    def test_reasoning_content_is_public_field(self) -> None:
        """Test that reasoning_content is accessible as a public attribute.

        FAILS before fix: reasoning_content doesn't exist as a field.
        PASSES after fix: reasoning_content is a documented public field.
        """
        chunk = AIMessageChunk(
            content="Hello",
            reasoning_content="Let me think about this...",
        )

        # Public API access
        assert chunk.reasoning_content == "Let me think about this..."
        assert hasattr(chunk, "reasoning_content")

    def test_reasoning_content_defaults_to_none(self) -> None:
        """Test that reasoning_content is None by default."""
        chunk = AIMessageChunk(content="Hello")

        assert chunk.reasoning_content is None

    def test_reasoning_only_chunk(self) -> None:
        """Test chunk with only reasoning, no content.

        This is the core scenario: models stream reasoning for 30+ seconds
        before any content appears.
        """
        chunk = AIMessageChunk(
            content="",
            reasoning_content="Step 1: Analyze the problem...",
        )

        assert chunk.content == ""
        assert chunk.reasoning_content == "Step 1: Analyze the problem..."
        # Chunk is not truly empty - it has semantic meaning via reasoning
        assert chunk.reasoning_content is not None

    def test_both_content_and_reasoning(self) -> None:
        """Test chunk with both content and reasoning."""
        chunk = AIMessageChunk(
            content="The answer is 42",
            reasoning_content="First, I calculated...",
        )

        assert chunk.content == "The answer is 42"
        assert chunk.reasoning_content == "First, I calculated..."


class TestReasoningContentMerging:
    """Test that reasoning_content is properly merged when chunks are combined."""

    def test_reasoning_concatenates_on_merge(self) -> None:
        """Test that reasoning_content is concatenated like content.

        FAILS before fix: reasoning_content is lost or overwritten.
        PASSES after fix: reasoning_content accumulates across chunks.
        """
        chunk1 = AIMessageChunk(content="", reasoning_content="Step 1: ")
        chunk2 = AIMessageChunk(content="", reasoning_content="Step 2: ")
        chunk3 = AIMessageChunk(content="Answer", reasoning_content="Step 3: ")

        merged = chunk1 + chunk2 + chunk3

        assert merged.reasoning_content == "Step 1: Step 2: Step 3: "
        assert merged.content == "Answer"

    def test_reasoning_and_content_both_accumulate(self) -> None:
        """Test that both content and reasoning accumulate independently."""
        chunk1 = AIMessageChunk(content="Hello", reasoning_content="Greeting... ")
        chunk2 = AIMessageChunk(content=" world", reasoning_content="Continuing... ")

        merged = chunk1 + chunk2

        assert merged.content == "Hello world"
        assert merged.reasoning_content == "Greeting... Continuing... "

    def test_none_reasoning_handled_correctly(self) -> None:
        """Test that None reasoning_content doesn't break merging."""
        chunk1 = AIMessageChunk(content="Hello", reasoning_content="Thinking... ")
        chunk2 = AIMessageChunk(content=" world", reasoning_content=None)

        merged = chunk1 + chunk2

        assert merged.content == "Hello world"
        assert merged.reasoning_content == "Thinking... "

    def test_both_none_reasoning(self) -> None:
        """Test merging chunks with no reasoning."""
        chunk1 = AIMessageChunk(content="Hello", reasoning_content=None)
        chunk2 = AIMessageChunk(content=" world", reasoning_content=None)

        merged = chunk1 + chunk2

        assert merged.content == "Hello world"
        assert merged.reasoning_content is None

    def test_multiple_reasoning_chunks_then_content(self) -> None:
        """Test realistic scenario: many reasoning chunks, then content.

        Simulates DeepSeek streaming where reasoning streams for 30+ seconds
        before content appears.
        """
        # 10 reasoning-only chunks
        reasoning_chunks = [
            AIMessageChunk(content="", reasoning_content=f"[token {i}] ")
            for i in range(10)
        ]

        # Final content chunk
        content_chunk = AIMessageChunk(content="Final answer", reasoning_content=None)

        # Merge all
        merged = reasoning_chunks[0]
        for chunk in reasoning_chunks[1:] + [content_chunk]:
            merged = merged + chunk

        # All reasoning accumulated
        reasoning = merged.reasoning_content or ""
        assert "[token 0]" in reasoning
        assert "[token 9]" in reasoning
        # Content present
        assert merged.content == "Final answer"

    def test_add_ai_message_chunks_function(self) -> None:
        """Test the add_ai_message_chunks function directly."""
        chunk1 = AIMessageChunk(content="A", reasoning_content="R1")
        chunk2 = AIMessageChunk(content="B", reasoning_content="R2")
        chunk3 = AIMessageChunk(content="C", reasoning_content="R3")

        merged = add_ai_message_chunks(chunk1, chunk2, chunk3)

        assert merged.content == "ABC"
        assert merged.reasoning_content == "R1R2R3"


class TestReasoningWithOtherFields:
    """Test that reasoning_content works correctly with other fields."""

    def test_reasoning_with_tool_calls(self) -> None:
        """Test that reasoning and tool calls can coexist."""
        from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk

        chunk = AIMessageChunk(
            content="",
            reasoning_content="Need to search...",
            tool_call_chunks=[
                create_tool_call_chunk(
                    name="search",
                    args='{"q": "test"}',
                    id="call_123",
                    index=0,
                )
            ],
        )

        assert chunk.reasoning_content == "Need to search..."
        assert len(chunk.tool_call_chunks) == 1

    def test_reasoning_with_usage_metadata(self) -> None:
        """Test that reasoning works with usage metadata."""
        from langchain_core.messages.ai import UsageMetadata

        chunk = AIMessageChunk(
            content="Answer",
            reasoning_content="Thinking...",
            usage_metadata=UsageMetadata(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
            ),
        )

        assert chunk.reasoning_content == "Thinking..."
        assert chunk.usage_metadata is not None
        assert chunk.usage_metadata["total_tokens"] == 30

    def test_reasoning_with_response_metadata(self) -> None:
        """Test that reasoning works with response metadata."""
        chunk = AIMessageChunk(
            content="Answer",
            reasoning_content="Thinking...",
            response_metadata={"model_provider": "deepseek"},
        )

        assert chunk.reasoning_content == "Thinking..."
        assert chunk.response_metadata["model_provider"] == "deepseek"


class TestBackwardCompatibility:
    """Test that non-reasoning chunks work identically."""

    def test_chunks_without_reasoning_unchanged(self) -> None:
        """Test that chunks without reasoning work as before."""
        chunk = AIMessageChunk(content="Hello world")

        assert chunk.content == "Hello world"
        assert chunk.reasoning_content is None

    def test_merging_without_reasoning_unchanged(self) -> None:
        """Test that merging non-reasoning chunks works as before."""
        chunk1 = AIMessageChunk(content="Hello")
        chunk2 = AIMessageChunk(content=" world")

        merged = chunk1 + chunk2

        assert merged.content == "Hello world"
        assert merged.reasoning_content is None

    def test_serialization_with_none_reasoning(self) -> None:
        """Test that serialization works with None reasoning_content."""
        chunk = AIMessageChunk(content="Hello", reasoning_content=None)

        # Should be able to create dict representation
        chunk_dict = chunk.model_dump()
        assert "reasoning_content" in chunk_dict
        assert chunk_dict["reasoning_content"] is None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_string_reasoning(self) -> None:
        """Test that empty string reasoning is handled correctly."""
        chunk = AIMessageChunk(content="Hello", reasoning_content="")

        # Empty string is valid (different from None)
        assert chunk.reasoning_content == ""

    def test_merge_empty_string_reasoning(self) -> None:
        """Test merging with empty string reasoning."""
        chunk1 = AIMessageChunk(content="A", reasoning_content="")
        chunk2 = AIMessageChunk(content="B", reasoning_content="Text")

        merged = chunk1 + chunk2

        # Empty string + "Text" = "Text"
        assert merged.reasoning_content == "Text"

    def test_long_reasoning_content(self) -> None:
        """Test that long reasoning content is handled correctly."""
        long_reasoning = "x" * 10000
        chunk = AIMessageChunk(content="Answer", reasoning_content=long_reasoning)

        assert len(chunk.reasoning_content or "") == 10000

    def test_unicode_in_reasoning(self) -> None:
        """Test that Unicode characters in reasoning work correctly."""
        chunk = AIMessageChunk(
            content="Answer",
            reasoning_content="思考中... 🤔 Thinking...",
        )

        assert "思考中" in (chunk.reasoning_content or "")
        assert "🤔" in (chunk.reasoning_content or "")

    def test_newlines_in_reasoning(self) -> None:
        """Test that newlines in reasoning are preserved."""
        chunk = AIMessageChunk(
            content="Answer",
            reasoning_content="Step 1:\nAnalyze\n\nStep 2:\nSolve",
        )

        assert "\n" in (chunk.reasoning_content or "")
        assert chunk.reasoning_content == "Step 1:\nAnalyze\n\nStep 2:\nSolve"
