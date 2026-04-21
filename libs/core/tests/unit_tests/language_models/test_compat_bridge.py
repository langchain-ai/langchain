"""Tests for the compat bridge (chunk-to-event conversion)."""

from typing import TYPE_CHECKING, cast

import pytest

from langchain_core.language_models._compat_bridge import (
    CompatBlock,
    _finalize_block,
    _normalize_finish_reason,
    _to_protocol_usage,
    amessage_to_events,
    chunks_to_events,
    message_to_events,
)
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

if TYPE_CHECKING:
    from langchain_protocol.protocol import (
        ContentBlockDeltaData,
        InvalidToolCallBlock,
        MessageFinishData,
        MessageStartData,
        ReasoningBlock,
        ServerToolCallBlock,
        TextBlock,
        ToolCallBlock,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_finalize_block_text_passes_through() -> None:
    block: CompatBlock = {"type": "text", "text": "hello"}
    result = _finalize_block(block)
    text_result = cast("TextBlock", result)
    assert text_result["type"] == "text"
    assert text_result["text"] == "hello"


def test_finalize_block_tool_call_chunk_valid_json() -> None:
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


def test_finalize_block_server_tool_call_chunk_valid_json() -> None:
    block: CompatBlock = {
        "type": "server_tool_call_chunk",
        "args": '{"q": "weather"}',
        "id": "srv_1",
        "name": "web_search",
    }
    result = _finalize_block(block)
    server_result = cast("ServerToolCallBlock", result)
    assert server_result["type"] == "server_tool_call"
    assert server_result["id"] == "srv_1"
    assert server_result["name"] == "web_search"
    assert server_result["args"] == {"q": "weather"}


def test_finalize_block_server_tool_call_chunk_invalid_json() -> None:
    block: CompatBlock = {
        "type": "server_tool_call_chunk",
        "args": "not json",
        "id": "srv_1",
        "name": "web_search",
    }
    result = _finalize_block(block)
    invalid = cast("InvalidToolCallBlock", result)
    assert invalid["type"] == "invalid_tool_call"
    assert invalid.get("error") is not None


def test_normalize_finish_reason() -> None:
    assert _normalize_finish_reason("stop") == "stop"
    assert _normalize_finish_reason("end_turn") == "stop"
    assert _normalize_finish_reason("length") == "length"
    assert _normalize_finish_reason("tool_use") == "tool_use"
    assert _normalize_finish_reason("tool_calls") == "tool_use"
    assert _normalize_finish_reason("content_filter") == "content_filter"
    assert _normalize_finish_reason(None) == "stop"


def test_to_protocol_usage_present() -> None:
    usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    result = _to_protocol_usage(usage)
    assert result is not None
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 20


def test_to_protocol_usage_none() -> None:
    assert _to_protocol_usage(None) is None


# ---------------------------------------------------------------------------
# chunks_to_events: streaming lifecycle
# ---------------------------------------------------------------------------


def test_chunks_to_events_text_only() -> None:
    """Multi-chunk text stream produces a clean lifecycle."""
    chunks = [
        ChatGenerationChunk(message=AIMessageChunk(content="Hello", id="msg-1")),
        ChatGenerationChunk(message=AIMessageChunk(content=" world", id="msg-1")),
    ]

    events = list(chunks_to_events(iter(chunks), message_id="msg-1"))
    event_types = [e["event"] for e in events]

    assert event_types[0] == "message-start"
    assert "content-block-start" in event_types
    assert event_types.count("content-block-delta") == 2
    assert "content-block-finish" in event_types
    assert event_types[-1] == "message-finish"

    finish = cast("MessageFinishData", events[-1])
    assert finish["reason"] == "stop"


def test_chunks_to_events_empty_iterator() -> None:
    """No chunks means no events."""
    assert list(chunks_to_events(iter([]))) == []


def test_chunks_to_events_block_transitions_close_previous_block() -> None:
    """String-keyed blocks that transition mid-stream each get their own lifecycle.

    Regression test for OpenAI `responses/v1` style streams where
    `content_blocks` uses string identifiers (e.g. `"lc_rs_305f30"`) to
    distinguish blocks. Each distinct block must get its own
    `content-block-start` / `content-block-finish` pair, with sequential
    `uint` wire indices, and blocks must not interleave.
    """
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {"type": "reasoning", "reasoning": "hmm", "index": "rs_a"},
                ],
                id="msg-1",
            )
        ),
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {"type": "reasoning", "reasoning": " then", "index": "rs_a"},
                ],
                id="msg-1",
            )
        ),
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {"type": "reasoning", "reasoning": "different", "index": "rs_b"},
                ],
                id="msg-1",
            )
        ),
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {"type": "text", "text": "answer: ", "index": "txt_1"},
                ],
                id="msg-1",
            )
        ),
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {"type": "text", "text": "42", "index": "txt_1"},
                ],
                id="msg-1",
            )
        ),
    ]

    events = list(chunks_to_events(iter(chunks), message_id="msg-1"))

    starts = [e for e in events if e["event"] == "content-block-start"]
    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [s["content_block"]["type"] for s in starts] == [
        "reasoning",
        "reasoning",
        "text",
    ]
    assert [f["content_block"]["type"] for f in finishes] == [
        "reasoning",
        "reasoning",
        "text",
    ]
    # Wire indices are sequential uints regardless of source-side keys.
    assert [s["index"] for s in starts] == [0, 1, 2]
    assert [f["index"] for f in finishes] == [0, 1, 2]

    # Finish events must be interleaved with starts (no-interleave rule):
    # block 0 finishes before block 1 starts, etc.
    lifecycle = [
        (e["event"], e["index"])
        for e in events
        if e["event"] in ("content-block-start", "content-block-finish")
    ]
    assert lifecycle == [
        ("content-block-start", 0),
        ("content-block-finish", 0),
        ("content-block-start", 1),
        ("content-block-finish", 1),
        ("content-block-start", 2),
        ("content-block-finish", 2),
    ]

    # Each finish carries the accumulated content for its block.
    assert finishes[0]["content_block"]["reasoning"] == "hmm then"
    assert finishes[1]["content_block"]["reasoning"] == "different"
    assert finishes[2]["content_block"]["text"] == "answer: 42"


def test_chunks_to_events_tool_call_multichunk() -> None:
    """Partial tool-call args across chunks finalize to a single tool_call."""
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id="msg-1",
                tool_call_chunks=[
                    {
                        "index": 0,
                        "id": "tc1",
                        "name": "search",
                        "args": '{"q":',
                        "type": "tool_call_chunk",
                    }
                ],
            )
        ),
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id="msg-1",
                tool_call_chunks=[
                    {
                        "index": 0,
                        "id": None,
                        "name": None,
                        "args": ' "test"}',
                        "type": "tool_call_chunk",
                    }
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

    # Exactly one block finalized, args parsed to a dict.
    finish_events = [e for e in events if e["event"] == "content-block-finish"]
    assert len(finish_events) == 1
    finalized = cast("ToolCallBlock", finish_events[0]["content_block"])
    assert finalized["type"] == "tool_call"
    assert finalized["args"] == {"q": "test"}

    # Valid tool_call at finish => finish_reason flips to tool_use.
    assert cast("MessageFinishData", events[-1])["reason"] == "tool_use"


def test_chunks_to_events_invalid_tool_call_keeps_stop_reason() -> None:
    """Malformed tool-args become invalid_tool_call; finish_reason stays `stop`."""
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id="msg-bad",
                tool_call_chunks=[
                    {
                        "index": 0,
                        "id": "tc1",
                        "name": "search",
                        "args": "{oops",
                        "type": "tool_call_chunk",
                    },
                ],
            )
        ),
    ]

    events = list(chunks_to_events(iter(chunks), message_id="msg-bad"))

    finish_events = [e for e in events if e["event"] == "content-block-finish"]
    assert len(finish_events) == 1
    assert finish_events[0]["content_block"]["type"] == "invalid_tool_call"
    assert cast("MessageFinishData", events[-1])["reason"] == "stop"


def test_chunks_to_events_anthropic_server_tool_use_routes_through_translator() -> None:
    """`server_tool_use` shape + anthropic provider tag becomes `server_tool_call`."""
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[
                    {"type": "text", "text": "Let me search. "},
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_01",
                        "name": "web_search",
                        "input": {"query": "weather"},
                    },
                ],
                response_metadata={"model_provider": "anthropic"},
            )
        ),
    ]

    events = list(chunks_to_events(iter(chunks)))
    finish_blocks = [
        e["content_block"] for e in events if e["event"] == "content-block-finish"
    ]
    block_types = [b.get("type") for b in finish_blocks]
    assert "server_tool_call" in block_types
    assert "text" in block_types


def test_chunks_to_events_unregistered_provider_falls_back() -> None:
    """Unknown provider tag doesn't crash; best-effort parsing surfaces text."""
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="Hello",
                response_metadata={"model_provider": "totally-made-up-provider"},
            )
        ),
    ]

    events = list(chunks_to_events(iter(chunks)))
    finish_events = [e for e in events if e["event"] == "content-block-finish"]
    assert [e["content_block"]["type"] for e in finish_events] == ["text"]


def test_chunks_to_events_no_provider_text_plus_tool_call() -> None:
    """Without a provider tag, text + tool_call_chunks both come through.

    This is the case the old legacy path silently dropped the tool call
    because it re-mined tool_call_chunks on top of the positional index
    already used by the text block. Trusting content_blocks keeps them
    on distinct indices.
    """
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="Hello",
                tool_call_chunks=[
                    {
                        "index": 1,
                        "id": "t1",
                        "name": "search",
                        "args": '{"q": "x"}',
                        "type": "tool_call_chunk",
                    },
                ],
            )
        ),
    ]

    events = list(chunks_to_events(iter(chunks)))
    finish_blocks = [
        e["content_block"] for e in events if e["event"] == "content-block-finish"
    ]
    types = [b.get("type") for b in finish_blocks]
    assert "text" in types
    assert "tool_call" in types


def test_chunks_to_events_reasoning_in_additional_kwargs() -> None:
    """Reasoning packed into additional_kwargs surfaces as a reasoning block."""
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[{"type": "text", "text": "2+2=4"}],
                additional_kwargs={"reasoning_content": "Adding two and two..."},
                response_metadata={"model_provider": "unknown-open-model"},
            )
        ),
    ]

    events = list(chunks_to_events(iter(chunks)))
    finish_blocks = [
        e["content_block"] for e in events if e["event"] == "content-block-finish"
    ]
    types = [b.get("type") for b in finish_blocks]
    assert "reasoning" in types
    assert "text" in types


# ---------------------------------------------------------------------------
# message_to_events: finalized-message replay
# ---------------------------------------------------------------------------


def test_message_to_events_text_only() -> None:
    msg = AIMessage(content="Hello world", id="msg-1")
    events = list(message_to_events(msg))

    event_types = [e["event"] for e in events]
    assert event_types == [
        "message-start",
        "content-block-start",
        "content-block-delta",
        "content-block-finish",
        "message-finish",
    ]
    start = cast("MessageStartData", events[0])
    assert start["message_id"] == "msg-1"

    delta_event = cast("ContentBlockDeltaData", events[2])
    delta = cast("TextBlock", delta_event["content_block"])
    assert delta["text"] == "Hello world"

    final = cast("MessageFinishData", events[-1])
    assert final["reason"] == "stop"


def test_message_to_events_empty_content_yields_start_finish_only() -> None:
    msg = AIMessage(content="", id="msg-empty")
    events = list(message_to_events(msg))
    event_types = [e["event"] for e in events]
    assert event_types == ["message-start", "message-finish"]


def test_message_to_events_reasoning_text_order() -> None:
    msg = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "think hard"},
            {"type": "text", "text": "the answer"},
        ],
        id="msg-2",
    )
    events = list(message_to_events(msg))

    starts = [e for e in events if e["event"] == "content-block-start"]
    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [s["content_block"]["type"] for s in starts] == ["reasoning", "text"]
    assert [f["content_block"]["type"] for f in finishes] == ["reasoning", "text"]

    deltas = [e for e in events if e["event"] == "content-block-delta"]
    assert len(deltas) == 2
    assert cast("ReasoningBlock", deltas[0]["content_block"])["reasoning"] == (
        "think hard"
    )
    assert cast("TextBlock", deltas[1]["content_block"])["text"] == "the answer"


def test_message_to_events_tool_call_skips_delta_and_infers_tool_use() -> None:
    msg = AIMessage(
        content="",
        id="msg-3",
        tool_calls=[
            {"id": "tc1", "name": "search", "args": {"q": "hi"}, "type": "tool_call"},
        ],
    )
    events = list(message_to_events(msg))

    # Finalized tool_call blocks carry no useful incremental text,
    # so no content-block-delta is emitted.
    deltas = [e for e in events if e["event"] == "content-block-delta"]
    assert deltas == []

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert len(finishes) == 1
    tc = cast("ToolCallBlock", finishes[0]["content_block"])
    assert tc["type"] == "tool_call"
    assert tc["args"] == {"q": "hi"}

    final = cast("MessageFinishData", events[-1])
    assert final["reason"] == "tool_use"


def test_message_to_events_invalid_tool_calls_surfaced_from_field() -> None:
    """`invalid_tool_calls` on AIMessage surface as protocol blocks.

    `AIMessage.content_blocks` does not currently include
    `invalid_tool_calls`, so the bridge merges them in explicitly.
    """
    msg = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "id": "call_1",
                "name": "search",
                "args": '{"q":',
                "error": "bad json",
            }
        ],
    )
    events = list(message_to_events(msg))
    finishes = [e for e in events if e["event"] == "content-block-finish"]
    types = [f["content_block"]["type"] for f in finishes]
    assert "invalid_tool_call" in types


def test_message_to_events_preserves_finish_reason_and_metadata() -> None:
    msg = AIMessage(
        content="done",
        id="msg-4",
        response_metadata={
            "finish_reason": "length",
            "model_name": "test-model",
            "stop_sequence": "</end>",
        },
    )
    events = list(message_to_events(msg))

    start = cast("MessageStartData", events[0])
    assert start["metadata"] == {"model": "test-model"}

    final = cast("MessageFinishData", events[-1])
    assert final["reason"] == "length"
    # finish_reason stripped from metadata; stop_sequence preserved
    assert final["metadata"] == {"model_name": "test-model", "stop_sequence": "</end>"}


def test_message_to_events_propagates_usage() -> None:
    msg = AIMessage(
        content="hi",
        id="msg-5",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
    events = list(message_to_events(msg))

    final = cast("MessageFinishData", events[-1])
    assert final["usage"] == {
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
    }


def test_message_to_events_message_id_override() -> None:
    msg = AIMessage(content="x", id="msg-orig")
    events = list(message_to_events(msg, message_id="msg-override"))
    start = cast("MessageStartData", events[0])
    assert start["message_id"] == "msg-override"


def test_message_to_events_self_contained_start_strips_heavy_fields() -> None:
    """`content-block-start` must not duplicate heavy payload fields.

    For image/audio/video/file/non_standard and finalized tool_call blocks,
    the large payload (base64 `data`, parsed `args`, arbitrary `value`)
    should appear only on `content-block-finish`, not on `content-block-start`.
    Start preserves correlation and small metadata fields.
    """
    msg = AIMessage(
        content=[
            {
                "type": "image",
                "id": "img-1",
                "mime_type": "image/png",
                "data": "A" * 1024,
            },
            {
                "type": "audio",
                "id": "aud-1",
                "mime_type": "audio/mp3",
                "data": "B" * 1024,
                "transcript": "hello",
            },
            {
                "type": "non_standard",
                "id": "ns-1",
                "value": {"big": "C" * 1024},
            },
        ],
        id="msg-heavy",
    )
    events = list(message_to_events(msg))

    starts = [e for e in events if e["event"] == "content-block-start"]
    assert [s["content_block"]["type"] for s in starts] == [
        "image",
        "audio",
        "non_standard",
    ]

    image_start = starts[0]["content_block"]
    assert image_start["id"] == "img-1"
    assert image_start["mime_type"] == "image/png"
    assert "data" not in image_start

    audio_start = starts[1]["content_block"]
    assert audio_start["id"] == "aud-1"
    assert audio_start["mime_type"] == "audio/mp3"
    assert "data" not in audio_start
    assert "transcript" not in audio_start

    ns_start = starts[2]["content_block"]
    assert ns_start["type"] == "non_standard"
    assert ns_start["value"] == {}

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert finishes[0]["content_block"]["data"] == "A" * 1024
    assert finishes[1]["content_block"]["data"] == "B" * 1024
    assert finishes[1]["content_block"]["transcript"] == "hello"
    assert finishes[2]["content_block"]["value"] == {"big": "C" * 1024}


def test_message_to_events_finalized_tool_call_start_strips_args() -> None:
    """Finalized `tool_call` keeps id/name on start but not parsed args."""
    msg = AIMessage(
        content="",
        id="msg-tc",
        tool_calls=[
            {
                "id": "tc1",
                "name": "search",
                "args": {"q": "big payload " * 100},
                "type": "tool_call",
            },
        ],
    )
    events = list(message_to_events(msg))

    starts = [e for e in events if e["event"] == "content-block-start"]
    assert len(starts) == 1
    tc_start = starts[0]["content_block"]
    assert tc_start["type"] == "tool_call"
    assert tc_start["id"] == "tc1"
    assert tc_start["name"] == "search"
    assert tc_start["args"] == {}

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    tc_finish = cast("ToolCallBlock", finishes[0]["content_block"])
    assert tc_finish["args"] == {"q": "big payload " * 100}


@pytest.mark.asyncio
async def test_amessage_to_events_matches_sync() -> None:
    msg = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "why"},
            {"type": "text", "text": "because"},
        ],
        id="msg-async",
    )
    sync_events = list(message_to_events(msg))
    async_events = [e async for e in amessage_to_events(msg)]
    assert async_events == sync_events
