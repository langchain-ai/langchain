"""Unit tests for reasoning extraction in OpenAI adapter.

These tests verify that the OpenAI adapter correctly extracts reasoning tokens
from vendor-specific delta fields and normalizes them into the first-class
reasoning_content field on AIMessageChunk.

Supported vendors:
- DeepSeek (reasoning_content field)
- Groq (reasoning field)
- xAI (reasoning_content field)
- OpenAI o1/o3 (reasoning field)
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessageChunk

from langchain_openai.chat_models.base import _convert_delta_to_message_chunk


class TestReasoningExtraction:
    """Test that reasoning is extracted from delta and placed in reasoning_content."""

    def test_extract_reasoning_content_from_deepseek(self) -> None:
        """Test extraction of reasoning_content field (DeepSeek, xAI).

        FAILS before fix: reasoning_content ignored, chunk.reasoning_content is None.
        PASSES after fix: chunk.reasoning_content contains the reasoning.
        """
        delta = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Let me think step by step...",
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert isinstance(chunk, AIMessageChunk)
        # First-class field (public API)
        assert chunk.reasoning_content == "Let me think step by step..."
        # Also in additional_kwargs for backward compat
        assert chunk.additional_kwargs.get("reasoning_content") == "Let me think step by step..."

    def test_extract_reasoning_from_groq(self) -> None:
        """Test extraction of reasoning field (Groq, OpenAI o1/o3)."""
        delta = {
            "role": "assistant",
            "content": "",
            "reasoning": "Analyzing the problem...",
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert isinstance(chunk, AIMessageChunk)
        # Normalized to reasoning_content
        assert chunk.reasoning_content == "Analyzing the problem..."
        # Also in additional_kwargs for backward compat
        assert chunk.additional_kwargs.get("reasoning") == "Analyzing the problem..."

    def test_both_content_and_reasoning(self) -> None:
        """Test that both content and reasoning are extracted."""
        delta = {
            "role": "assistant",
            "content": "The answer is 42",
            "reasoning_content": "First, I calculated...",
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert chunk.content == "The answer is 42"
        assert chunk.reasoning_content == "First, I calculated..."

    def test_no_reasoning_fields(self) -> None:
        """Test that non-reasoning deltas work unchanged."""
        delta = {
            "role": "assistant",
            "content": "Hello, world!",
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert chunk.content == "Hello, world!"
        assert chunk.reasoning_content is None

    def test_reasoning_only_chunk_not_empty(self) -> None:
        """Test that chunks with only reasoning are not semantically empty.

        This is the core bug fix: preventing 30+ seconds of empty chunks.
        """
        delta = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Step 1: Analyze...",
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        # Content is empty string
        assert chunk.content == ""
        # But chunk has semantic meaning via reasoning
        assert chunk.reasoning_content == "Step 1: Analyze..."
        assert chunk.reasoning_content is not None


class TestVendorNormalization:
    """Test that different vendor field names are normalized."""

    def test_reasoning_content_field_normalized(self) -> None:
        """Test that reasoning_content is normalized to reasoning_content."""
        delta = {"role": "assistant", "reasoning_content": "Thinking..."}
        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert chunk.reasoning_content == "Thinking..."

    def test_reasoning_field_normalized(self) -> None:
        """Test that reasoning is normalized to reasoning_content."""
        delta = {"role": "assistant", "reasoning": "Thinking..."}
        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        # Normalized to reasoning_content
        assert chunk.reasoning_content == "Thinking..."

    def test_reasoning_content_takes_precedence(self) -> None:
        """Test that reasoning_content takes precedence over reasoning.

        If both fields are present (unlikely but possible), reasoning_content wins.
        """
        delta = {
            "role": "assistant",
            "reasoning_content": "From reasoning_content",
            "reasoning": "From reasoning",
        }
        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        # reasoning_content takes precedence
        assert chunk.reasoning_content == "From reasoning_content"


class TestReasoningWithOtherFields:
    """Test that reasoning extraction works with other delta fields."""

    def test_reasoning_with_function_call(self) -> None:
        """Test that reasoning and function_call can coexist."""
        delta = {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "get_weather", "arguments": '{"city": "SF"}'},
            "reasoning_content": "User wants weather info...",
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert chunk.reasoning_content == "User wants weather info..."
        assert "function_call" in chunk.additional_kwargs

    def test_reasoning_with_tool_calls(self) -> None:
        """Test that reasoning and tool_calls can coexist."""
        delta = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_123",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                }
            ],
            "reasoning": "Need to search...",
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert chunk.reasoning_content == "Need to search..."
        assert len(chunk.tool_call_chunks) == 1


class TestBackwardCompatibility:
    """Test that backward compatibility is preserved."""

    def test_additional_kwargs_still_populated(self) -> None:
        """Test that reasoning is also in additional_kwargs for backward compat.

        Users who were accessing reasoning via additional_kwargs should still work.
        """
        delta = {
            "role": "assistant",
            "reasoning_content": "Thinking...",
        }
        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        # First-class field (new public API)
        assert chunk.reasoning_content == "Thinking..."
        # Also in additional_kwargs (backward compat)
        assert chunk.additional_kwargs.get("reasoning_content") == "Thinking..."

    def test_non_reasoning_models_unchanged(self) -> None:
        """Test that non-reasoning models work identically."""
        delta = {
            "role": "assistant",
            "content": "Hello",
        }
        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert chunk.content == "Hello"
        assert chunk.reasoning_content is None
        # No reasoning in additional_kwargs
        assert "reasoning" not in chunk.additional_kwargs
        assert "reasoning_content" not in chunk.additional_kwargs

    def test_human_message_unchanged(self) -> None:
        """Test that human messages are unaffected."""
        delta = {
            "role": "user",
            "content": "Hello",
            "reasoning_content": "This should be ignored",
        }

        chunk = _convert_delta_to_message_chunk(delta, HumanMessageChunk)

        assert isinstance(chunk, HumanMessageChunk)
        assert chunk.content == "Hello"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_reasoning_content_ignored(self) -> None:
        """Test that None reasoning_content is handled gracefully."""
        delta = {
            "role": "assistant",
            "content": "Hello",
            "reasoning_content": None,
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert chunk.content == "Hello"
        assert chunk.reasoning_content is None

    def test_empty_string_reasoning_content(self) -> None:
        """Test that empty string reasoning_content is ignored.

        Empty string is falsy in Python, so it won't be extracted.
        """
        delta = {
            "role": "assistant",
            "content": "Hello",
            "reasoning_content": "",
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert chunk.content == "Hello"
        # Empty string is falsy, so not extracted
        assert chunk.reasoning_content is None

    def test_reasoning_with_none_content(self) -> None:
        """Test that None content is handled correctly with reasoning."""
        delta = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Thinking...",
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert chunk.content == ""  # None becomes empty string
        assert chunk.reasoning_content == "Thinking..."

    def test_unicode_in_reasoning(self) -> None:
        """Test that Unicode in reasoning is handled correctly."""
        delta = {
            "role": "assistant",
            "reasoning_content": "思考中... 🤔",
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert chunk.reasoning_content == "思考中... 🤔"

    def test_long_reasoning_content(self) -> None:
        """Test that long reasoning content is handled correctly."""
        long_reasoning = "x" * 10000
        delta = {
            "role": "assistant",
            "reasoning_content": long_reasoning,
        }

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)

        assert len(chunk.reasoning_content or "") == 10000
