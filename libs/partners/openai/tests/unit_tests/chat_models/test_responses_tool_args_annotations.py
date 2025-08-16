"""Test for responses streaming tool call with annotations issue (#32562)."""

from typing import Any, Optional
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessageChunk
from pydantic import SecretStr

from langchain_openai import ChatOpenAI
from tests.unit_tests.chat_models.test_base import MockSyncContextManager


class MockChunk:
    """Mock chunk for responses streaming events."""

    def __init__(self, chunk_type: str, **kwargs: Any) -> None:
        self.type = chunk_type
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_tool_call_args_with_annotation_index_consistency() -> None:
    """Test tool call args consistent index when annotations are interleaved.

    This test reproduces issue #32562 where output_text.annotation.added events
    advance the sub-index causing function_call_arguments.delta chunks to get
    different indices, leading to invalid_tool_calls.
    """
    # Create a sequence of mock chunks that reproduces the bug scenario:
    # 1. Start a function call
    # 2. Stream some function arguments
    # 3. Interleave an annotation event (this advances sub-index in buggy version)
    # 4. Stream more function arguments (these get different index in buggy version)

    chunks_with_bug = [
        # Start function call at output_index=1
        MockChunk(
            "response.output_item.added",
            output_index=1,
            item=MockChunk(
                "function_call",
                type="function_call",
                name="test_function",
                arguments="",
                call_id="call_123",
                id="item_123",
            ),
        ),
        # First function argument delta
        MockChunk(
            "response.function_call_arguments.delta", output_index=1, delta='{"param'
        ),
        # Second function argument delta
        MockChunk(
            "response.function_call_arguments.delta", output_index=1, delta='": "valu'
        ),
        # Annotation event - this advances sub-index in buggy version
        MockChunk(
            "response.output_text.annotation.added",
            output_index=1,
            content_index=0,
            annotation={"type": "file_citation", "file_id": "file_123"},
        ),
        # More function argument deltas - these get different index in buggy version
        MockChunk(
            "response.function_call_arguments.delta", output_index=1, delta='e"}'
        ),
    ]

    llm = ChatOpenAI(
        model="o4-mini", output_version="responses/v1", api_key=SecretStr("fake-key")
    )
    mock_client = MagicMock()

    def mock_create(*args: Any, **kwargs: Any) -> MockSyncContextManager:
        return MockSyncContextManager(chunks_with_bug)

    mock_client.responses.create = mock_create

    # Stream and aggregate chunks
    chunks: list[AIMessageChunk] = []
    with patch.object(llm, "root_client", mock_client):
        for chunk in llm.stream("test"):
            assert isinstance(chunk, AIMessageChunk)
            chunks.append(chunk)

    # Merge all chunks to simulate complete message assembly
    full: Optional[AIMessageChunk] = None
    for chunk in chunks:
        if full is None:
            full = chunk
        else:
            result = full + chunk
            assert isinstance(result, AIMessageChunk)
            full = result

    assert isinstance(full, AIMessageChunk)

    # Check that all tool_call_chunks have the same index
    tool_call_indices = set()
    for chunk in chunks:
        for tool_chunk in chunk.tool_call_chunks:
            tool_call_indices.add(tool_chunk.get("index"))

    # All tool call chunks should have the same index
    assert len(tool_call_indices) == 1, (
        f"Tool call chunks have inconsistent indices: {tool_call_indices}"
    )

    # The final assembled message should have empty invalid_tool_calls
    assert hasattr(full, "invalid_tool_calls")
    assert full.invalid_tool_calls == [], (
        f"Found invalid_tool_calls: {full.invalid_tool_calls}"
    )

    # The final message should have one complete tool call with full args
    assert len(full.tool_calls) == 1
    tool_call = full.tool_calls[0]
    assert tool_call["name"] == "test_function"
    assert tool_call["id"] == "call_123"

    # The arguments should be complete and properly parsed
    args = tool_call["args"]
    assert {"param": "value"} == args, f"Expected complete args, got: {args}"
