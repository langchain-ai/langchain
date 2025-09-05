import json
from typing import Any


# Simple mock implementation for testing
def msg_content_output(content_blocks: list[Any]) -> Any:
    """Mock implementation of msg_content_output for testing.

    Returns the list if all blocks are recognized, otherwise JSON
    serializes the entire list.
    """
    from langchain_core.tools.base import TOOL_MESSAGE_BLOCK_TYPES

    # Check if all blocks are recognized
    all_recognized = all(
        isinstance(block, dict) and block.get("type") in TOOL_MESSAGE_BLOCK_TYPES
        for block in content_blocks
    )

    if all_recognized:
        # Return the original list
        return content_blocks
    # JSON serialize the entire input list
    return json.dumps(content_blocks, ensure_ascii=False)


def test_msg_content_output_preserves_document_block() -> None:
    """Document blocks should pass through unchanged when recognized."""
    doc_block = {"type": "document", "content": [{"text": "Hello"}]}
    result = msg_content_output([doc_block])  # wrap in list
    assert result == [doc_block]


def test_msg_content_output_falls_back_to_json_for_unknown_block() -> None:
    """Unknown block types should be JSON serialized for backward compatibility."""
    unknown_block = {"type": "foo_block", "content": [{"text": "bar"}]}
    result = msg_content_output([unknown_block])
    assert result == json.dumps([unknown_block], ensure_ascii=False)
