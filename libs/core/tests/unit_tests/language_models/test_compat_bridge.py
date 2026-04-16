"""Tests for the compat bridge (chunk-to-event conversion)."""

from typing import TYPE_CHECKING, cast

from langchain_core.language_models._compat_bridge import (
    CompatBlock,
    _accumulate_block,
    _delta_block,
    _extract_blocks_from_chunk,
    _extract_final_blocks,
    _finalize_block,
    _make_start_block,
    _normalize_finish_reason,
    _to_protocol_usage,
    chunks_to_events,
)
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

if TYPE_CHECKING:
    from langchain_protocol.protocol import (
        InvalidToolCallBlock,
        MessageFinishData,
        ReasoningBlock,
        TextBlock,
        ToolCallBlock,
    )


def test_accumulate_block_text() -> None:
    acc: CompatBlock = {"type": "text", "text": "Hello"}
    delta: CompatBlock = {"type": "text", "text": " world"}
    result = _accumulate_block(acc, delta)
    assert result["text"] == "Hello world"


def test_accumulate_block_reasoning() -> None:
    acc: CompatBlock = {"type": "reasoning", "reasoning": "think"}
    delta: CompatBlock = {"type": "reasoning", "reasoning": "ing"}
    result = _accumulate_block(acc, delta)
    assert result["reasoning"] == "thinking"


def test_accumulate_block_tool_call_chunk() -> None:
    acc: CompatBlock = {"type": "tool_call_chunk", "args": '{"na'}
    delta: CompatBlock = {
        "type": "tool_call_chunk",
        "args": 'me": "test"}',
        "id": "tc1",
    }
    result = _accumulate_block(acc, delta)
    assert result["args"] == '{"name": "test"}'
    assert result["id"] == "tc1"


def test_delta_block_text() -> None:
    prev: CompatBlock = {"type": "text", "text": "Hello"}
    cur: CompatBlock = {"type": "text", "text": "Hello world"}
    delta = _delta_block(prev, cur)
    assert delta is not None
    text_delta = cast("TextBlock", delta)
    assert text_delta["text"] == " world"


def test_delta_block_no_change() -> None:
    prev: CompatBlock = {"type": "text", "text": "Hello"}
    cur: CompatBlock = {"type": "text", "text": "Hello"}
    delta = _delta_block(prev, cur)
    assert delta is None


def test_delta_block_reasoning() -> None:
    prev: CompatBlock = {"type": "reasoning", "reasoning": "step 1"}
    cur: CompatBlock = {"type": "reasoning", "reasoning": "step 1, step 2"}
    delta = _delta_block(prev, cur)
    assert delta is not None
    reasoning_delta = cast("ReasoningBlock", delta)
    assert reasoning_delta["reasoning"] == ", step 2"


def test_finalize_block_text() -> None:
    block: CompatBlock = {"type": "text", "text": "hello"}
    result = _finalize_block(block)
    text_result = cast("TextBlock", result)
    assert text_result["type"] == "text"
    assert text_result["text"] == "hello"


def test_finalize_block_tool_call_chunk_valid() -> None:
    block: CompatBlock = {
        "type": "tool_call_chunk",
        "args": '{"query": "test"}',
        "id": "tc1",
        "name": "search",
    }
    result = _finalize_block(block)
    tool_call = cast("ToolCallBlock", result)
    assert tool_call["type"] == "tool_call"
    assert tool_call["id"] == "tc1"
    assert tool_call["name"] == "search"
    assert tool_call["args"] == {"query": "test"}


def test_finalize_block_tool_call_chunk_invalid_json() -> None:
    block: CompatBlock = {
        "type": "tool_call_chunk",
        "args": "not json",
        "id": "tc1",
        "name": "search",
    }
    result = _finalize_block(block)
    invalid = cast("InvalidToolCallBlock", result)
    assert invalid["type"] == "invalid_tool_call"
    assert invalid.get("error") is not None


def test_make_start_block_text() -> None:
    block: CompatBlock = {"type": "text", "text": "hello"}
    start = _make_start_block(block)
    text_start = cast("TextBlock", start)
    assert text_start["type"] == "text"
    assert text_start["text"] == ""


def test_make_start_block_reasoning() -> None:
    block: CompatBlock = {"type": "reasoning", "reasoning": "thinking"}
    start = _make_start_block(block)
    reasoning_start = cast("ReasoningBlock", start)
    assert reasoning_start["type"] == "reasoning"
    assert reasoning_start["reasoning"] == ""


def test_normalize_finish_reason() -> None:
    assert _normalize_finish_reason("stop") == "stop"
    assert _normalize_finish_reason("end_turn") == "stop"
    assert _normalize_finish_reason("length") == "length"
    assert _normalize_finish_reason("tool_use") == "tool_use"
    assert _normalize_finish_reason("tool_calls") == "tool_use"
    assert _normalize_finish_reason("content_filter") == "content_filter"
    assert _normalize_finish_reason(None) == "stop"


def test_to_protocol_usage() -> None:
    usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    result = _to_protocol_usage(usage)
    assert result is not None
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 20


def test_to_protocol_usage_none() -> None:
    assert _to_protocol_usage(None) is None


def test_extract_blocks_from_chunk_string_content() -> None:
    msg = AIMessageChunk(content="Hello")
    blocks = _extract_blocks_from_chunk(msg)
    assert len(blocks) == 1
    idx, block = blocks[0]
    assert idx == 0
    assert block["type"] == "text"
    assert block["text"] == "Hello"


def test_extract_blocks_from_chunk_list_content() -> None:
    msg = AIMessageChunk(
        content=[
            {"type": "text", "text": "Hello"},
            {"type": "reasoning", "reasoning": "think"},
        ]
    )
    blocks = _extract_blocks_from_chunk(msg)
    assert len(blocks) == 2
    assert blocks[0][1]["type"] == "text"
    assert blocks[1][1]["type"] == "reasoning"


def test_extract_blocks_from_chunk_tool_call_chunks() -> None:
    msg = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"index": 1, "id": "tc1", "name": "search", "args": '{"q": "test"}'}
        ],
    )
    blocks = _extract_blocks_from_chunk(msg)
    assert len(blocks) == 1
    idx, block = blocks[0]
    assert idx == 1
    assert block["type"] == "tool_call_chunk"
    assert block["id"] == "tc1"


def test_extract_final_blocks() -> None:
    msg = AIMessage(
        content="Hello",
        tool_calls=[{"id": "tc1", "name": "search", "args": {"q": "test"}}],
    )
    blocks = _extract_final_blocks(msg)
    assert len(blocks) == 2
    assert blocks[0][1]["type"] == "text"
    assert blocks[1][1]["type"] == "tool_call"


def test_chunks_to_events_text_only() -> None:
    """Test full lifecycle with text-only stream."""
    chunks = [
        ChatGenerationChunk(message=AIMessageChunk(content="Hello", id="msg-1")),
        ChatGenerationChunk(message=AIMessageChunk(content=" world", id="msg-1")),
    ]

    events = list(chunks_to_events(iter(chunks), message_id="msg-1"))

    # Expected: message-start, content-block-start, delta, delta,
    # content-block-finish, message-finish
    event_types = [e["event"] for e in events]
    assert event_types[0] == "message-start"
    assert "content-block-start" in event_types
    assert "content-block-delta" in event_types
    assert "content-block-finish" in event_types
    assert event_types[-1] == "message-finish"

    # Verify the finish event
    finish = cast("MessageFinishData", events[-1])
    assert finish["reason"] == "stop"


def test_chunks_to_events_empty() -> None:
    """No chunks means no events."""
    events = list(chunks_to_events(iter([])))
    assert events == []


def test_chunks_to_events_tool_call() -> None:
    """Test lifecycle with tool call chunks."""
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id="msg-1",
                tool_call_chunks=[
                    {"index": 0, "id": "tc1", "name": "search", "args": '{"q":'}
                ],
            )
        ),
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id="msg-1",
                tool_call_chunks=[
                    {"index": 0, "id": None, "name": None, "args": ' "test"}'}
                ],
            )
        ),
    ]

    events = list(chunks_to_events(iter(chunks), message_id="msg-1"))
    event_types = [e["event"] for e in events]

    assert event_types[0] == "message-start"
    assert "content-block-start" in event_types
    assert "content-block-finish" in event_types
    assert event_types[-1] == "message-finish"

    # Finish reason should be tool_use
    final = cast("MessageFinishData", events[-1])
    assert final["reason"] == "tool_use"

    # Find the content-block-finish event
    finish_events = [e for e in events if e["event"] == "content-block-finish"]
    assert len(finish_events) == 1
    finish_tc = cast("ToolCallBlock", finish_events[0]["content_block"])
    assert finish_tc["type"] == "tool_call"
    assert finish_tc["args"] == {"q": "test"}


def test_chunks_to_events_invalid_tool_call_finish_reason() -> None:
    """Malformed JSON tool args should NOT infer finish_reason="tool_use".

    Fix 5a from the streaming-v2 plan: the ``tool_use`` inference was
    previously triggered by any tool-like block including
    ``invalid_tool_call``. Now it should only fire for in-flight
    ``tool_call_chunk`` / finalized ``tool_call`` blocks.
    """
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id="msg-bad",
                tool_call_chunks=[
                    {"index": 0, "id": "tc1", "name": "search", "args": "{oops"},
                ],
            )
        ),
    ]

    events = list(chunks_to_events(iter(chunks), message_id="msg-bad"))

    # The finalized block at content-block-finish should be invalid_tool_call
    finish_events = [e for e in events if e["event"] == "content-block-finish"]
    assert len(finish_events) == 1
    finalized = finish_events[0]["content_block"]
    assert finalized["type"] == "invalid_tool_call"

    # And the message-finish reason must remain "stop" — we never had a
    # valid tool_call/tool_call_chunk at finish time, so no tool_use inference.
    final = cast("MessageFinishData", events[-1])
    assert final["reason"] == "stop"
