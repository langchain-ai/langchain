"""Tests for streaming output parser concurrency safety.

These tests verify that structured output parsing is safe under concurrent
execution and that streaming chunk aggregation does not corrupt data when
multiple invocations run in parallel.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessageChunk
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGenerationChunk


class TestSchema(BaseModel):
    """Test schema for structured output."""

    name: str = Field(description="The name field")
    value: int = Field(description="The value field")
    items: list[str] = Field(default_factory=list, description="List of items")


def create_streaming_chunks(
    tool_name: str, args_parts: list[str], tool_id: str = "test_id"
) -> list[AIMessageChunk]:
    """Create a sequence of streaming chunks that build up a tool call.

    Args:
        tool_name: Name of the tool being called.
        args_parts: List of argument string fragments to stream.
        tool_id: ID for the tool call.

    Returns:
        List of AIMessageChunk objects representing a streaming response.
    """
    chunks = []

    # First chunk with tool name
    chunks.append(
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                create_tool_call_chunk(
                    name=tool_name, args="", id=tool_id, index=0
                )
            ],
        )
    )

    # Subsequent chunks with argument fragments
    for args_part in args_parts:
        chunks.append(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    create_tool_call_chunk(
                        name=None, args=args_part, id=None, index=0
                    )
                ],
            )
        )

    return chunks


def simulate_streaming_parse(
    parser: JsonOutputKeyToolsParser | PydanticToolsParser,
    chunks: list[AIMessageChunk],
) -> Any:
    """Simulate streaming parsing by aggregating chunks and parsing incrementally.

    Args:
        parser: The output parser to use.
        chunks: List of message chunks to aggregate and parse.

    Returns:
        The final parsed result.

    Raises:
        Exception: If parsing fails at any point.
    """
    accumulated: AIMessageChunk | None = None

    for chunk in chunks:
        if accumulated is None:
            accumulated = chunk
        else:
            accumulated = accumulated + chunk

    # Parse the final accumulated result
    generation = ChatGenerationChunk(message=accumulated)
    return parser.parse_result([generation], partial=False)


class TestStreamingConcurrency:
    """Test suite for streaming output parser concurrency safety."""

    def test_sequential_streaming_baseline(self) -> None:
        """Baseline test: sequential streaming should always work."""
        parser = JsonOutputKeyToolsParser(
            key_name="TestSchema", first_tool_only=True
        )

        args_parts = ['{"name": "test', '", "value": ', "42}"]
        chunks = create_streaming_chunks("TestSchema", args_parts)

        result = simulate_streaming_parse(parser, chunks)

        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_concurrent_streaming_isolation(self) -> None:
        """Test that concurrent streaming invocations don't interfere.

        This test runs multiple streaming parses in parallel with different
        data to verify that chunk aggregation is properly isolated per
        invocation.
        """
        parser = JsonOutputKeyToolsParser(
            key_name="TestSchema", first_tool_only=True
        )

        def parse_stream(stream_id: int) -> dict[str, Any]:
            """Parse a stream with unique data based on stream_id."""
            args_parts = [
                f'{{"name": "stream_{stream_id}',
                f'", "value": {stream_id * 10}',
                "}",
            ]
            chunks = create_streaming_chunks("TestSchema", args_parts)
            return simulate_streaming_parse(parser, chunks)

        # Run 20 concurrent streaming operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(parse_stream, i) for i in range(20)]
            results = [f.result() for f in futures]

        # Verify each result matches its expected data
        for i, result in enumerate(results):
            assert result is not None, f"Stream {i} returned None"
            assert (
                result["name"] == f"stream_{i}"
            ), f"Stream {i} name mismatch: {result['name']}"
            assert (
                result["value"] == i * 10
            ), f"Stream {i} value mismatch: {result['value']}"

    @pytest.mark.asyncio
    async def test_async_concurrent_streaming_isolation(self) -> None:
        """Test async concurrent streaming invocations don't interfere."""
        parser = JsonOutputKeyToolsParser(
            key_name="TestSchema", first_tool_only=True
        )

        async def parse_stream_async(stream_id: int) -> dict[str, Any]:
            """Parse a stream asynchronously with unique data."""
            # Simulate async streaming with small delays
            await asyncio.sleep(0.001 * (stream_id % 5))

            args_parts = [
                f'{{"name": "async_{stream_id}',
                f'", "value": {stream_id * 100}',
                "}",
            ]
            chunks = create_streaming_chunks("TestSchema", args_parts)
            return simulate_streaming_parse(parser, chunks)

        # Run 20 concurrent async streaming operations
        tasks = [parse_stream_async(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # Verify each result matches its expected data
        for i, result in enumerate(results):
            assert result is not None, f"Async stream {i} returned None"
            assert (
                result["name"] == f"async_{i}"
            ), f"Async stream {i} name mismatch: {result['name']}"
            assert (
                result["value"] == i * 100
            ), f"Async stream {i} value mismatch: {result['value']}"

    def test_pydantic_parser_concurrent_streaming(self) -> None:
        """Test PydanticToolsParser under concurrent streaming."""
        parser = PydanticToolsParser(tools=[TestSchema], first_tool_only=True)

        def parse_stream(stream_id: int) -> TestSchema:
            """Parse a stream into a Pydantic model."""
            args_parts = [
                f'{{"name": "pydantic_{stream_id}',
                f'", "value": {stream_id}',
                ', "items": ["a", "b"]}',
            ]
            chunks = create_streaming_chunks("TestSchema", args_parts)
            return simulate_streaming_parse(parser, chunks)

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(parse_stream, i) for i in range(15)]
            results = [f.result() for f in futures]

        # Verify Pydantic models are correctly parsed
        for i, result in enumerate(results):
            assert isinstance(result, TestSchema)
            assert result.name == f"pydantic_{i}"
            assert result.value == i
            assert result.items == ["a", "b"]

    def test_interleaved_chunk_aggregation(self) -> None:
        """Test that chunk aggregation doesn't mix data from different streams.

        This test simulates a scenario where chunks from different streams
        might arrive in an interleaved fashion (though they should be
        processed independently).
        """
        parser = JsonOutputKeyToolsParser(
            key_name="TestSchema", first_tool_only=True
        )

        def parse_with_delay(stream_id: int, delay_ms: int) -> dict[str, Any]:
            """Parse with artificial delay to increase interleaving chance."""
            import time

            time.sleep(delay_ms / 1000.0)

            args_parts = [
                f'{{"name": "delayed_{stream_id}',
                f'", "value": {stream_id}',
                "}",
            ]
            chunks = create_streaming_chunks("TestSchema", args_parts)
            return simulate_streaming_parse(parser, chunks)

        # Run with varying delays to maximize interleaving
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(parse_with_delay, i, (i % 3) * 2)
                for i in range(10)
            ]
            results = [f.result() for f in futures]

        # Verify no data corruption occurred
        for i, result in enumerate(results):
            assert result["name"] == f"delayed_{i}"
            assert result["value"] == i

    def test_empty_arguments_handling(self) -> None:
        """Test that empty or None arguments are handled correctly."""
        parser = JsonOutputKeyToolsParser(
            key_name="TestSchema", first_tool_only=True
        )

        # Test with empty string arguments
        chunks = [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    create_tool_call_chunk(
                        name="TestSchema", args="", id="test_id", index=0
                    )
                ],
            )
        ]

        accumulated = chunks[0]
        generation = ChatGenerationChunk(message=accumulated)
        result = parser.parse_result([generation], partial=False)

        # Empty arguments should parse to empty dict
        assert result == {}

    def test_partial_json_parsing_safety(self) -> None:
        """Test that partial JSON parsing doesn't corrupt state."""
        parser = JsonOutputKeyToolsParser(
            key_name="TestSchema", first_tool_only=True
        )

        # Simulate partial streaming where JSON is incomplete
        partial_chunks = [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    create_tool_call_chunk(
                        name="TestSchema", args='{"name":', id="test_id", index=0
                    )
                ],
            )
        ]

        accumulated = partial_chunks[0]
        generation = ChatGenerationChunk(message=accumulated)

        # Partial parsing should return None or handle gracefully
        result = parser.parse_result([generation], partial=True)
        # Partial result may be None or incomplete dict
        assert result is None or isinstance(result, dict)


class TestChunkAggregationIsolation:
    """Test that chunk aggregation maintains per-invocation isolation."""

    def test_message_chunk_addition_creates_new_objects(self) -> None:
        """Verify that adding message chunks creates new objects, not mutations."""
        chunk1 = AIMessageChunk(
            content="Hello",
            tool_call_chunks=[
                create_tool_call_chunk(
                    name="tool1", args='{"a":', id="id1", index=0
                )
            ],
        )

        chunk2 = AIMessageChunk(
            content=" World",
            tool_call_chunks=[
                create_tool_call_chunk(name=None, args=' 1}', id=None, index=0)
            ],
        )

        # Add chunks
        result = chunk1 + chunk2

        # Verify original chunks are unchanged
        assert chunk1.content == "Hello"
        assert len(chunk1.tool_call_chunks) == 1
        assert chunk1.tool_call_chunks[0]["args"] == '{"a":'

        assert chunk2.content == " World"
        assert len(chunk2.tool_call_chunks) == 1
        assert chunk2.tool_call_chunks[0]["args"] == " 1}"

        # Verify result is a new object with combined data
        assert result.content == "Hello World"
        assert len(result.tool_call_chunks) == 1
        assert result.tool_call_chunks[0]["args"] == '{"a": 1}'

    def test_concurrent_chunk_aggregation_independence(self) -> None:
        """Test that concurrent chunk aggregations don't share state."""

        def aggregate_chunks(prefix: str) -> str:
            """Aggregate chunks with a specific prefix."""
            chunks = [
                AIMessageChunk(content=f"{prefix}_1"),
                AIMessageChunk(content=f"{prefix}_2"),
                AIMessageChunk(content=f"{prefix}_3"),
            ]

            result = chunks[0]
            for chunk in chunks[1:]:
                result = result + chunk

            return result.content

        # Run concurrent aggregations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(aggregate_chunks, f"stream{i}") for i in range(10)
            ]
            results = [f.result() for f in futures]

        # Verify each aggregation produced correct result
        for i, result in enumerate(results):
            expected = f"stream{i}_1stream{i}_2stream{i}_3"
            assert result == expected, f"Stream {i} aggregation corrupted: {result}"
