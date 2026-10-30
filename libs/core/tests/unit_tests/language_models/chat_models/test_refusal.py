"""Refusal chunk aggregation."""

from langchain_core.messages import AIMessageChunk


def test_anthropic_terminal_refusal_chunk() -> None:
    first = AIMessageChunk(
        content="blocked", response_metadata={"model_provider": "anthropic"}
    )
    last = AIMessageChunk(
        content="",
        response_metadata={"model_provider": "anthropic", "stop_reason": "refusal"},
        chunk_position="last",
    )
    assert first.content_blocks == [{"type": "text", "text": "blocked"}]
    assert any(block["type"] == "refusal" for block in last.content_blocks)
    assert (first + last).content_blocks == [{"type": "refusal", "refusal": "blocked"}]
