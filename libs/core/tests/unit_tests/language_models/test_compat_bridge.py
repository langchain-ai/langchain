"""Tests for the compat bridge (chunk-to-event conversion)."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_tests.utils.stream_lifecycle import assert_valid_event_stream

from langchain_core.language_models._compat_bridge import (
    CompatBlock,
    _finalize_block,
    _to_protocol_usage,
    achunks_to_events,
    amessage_to_events,
    chunks_to_events,
    message_to_events,
)
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

if TYPE_CHECKING:
    from langchain_protocol.protocol import (
        ContentBlockDeltaData,
        InvalidToolCall,
        MessageFinishData,
        MessageStartData,
        ServerToolCall,
        TextContentBlock,
        ToolCall,
    )


def _event_metadata(event: Any) -> dict[str, Any]:
    """Return event metadata for protocol versions that type it as extensible."""
    return cast("dict[str, Any]", cast("dict[str, Any]", event).get("metadata") or {})


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_finalize_block_text_passes_through() -> None:
    block: CompatBlock = {"type": "text", "text": "hello"}
    result = _finalize_block(block)
    text_result = cast("TextContentBlock", result)
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
    tool_call = cast("ToolCall", result)
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
    invalid = cast("InvalidToolCall", result)
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
    server_result = cast("ServerToolCall", result)
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
    invalid = cast("InvalidToolCall", result)
    assert invalid["type"] == "invalid_tool_call"
    assert invalid.get("error") is not None


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
    # No provider finish_reason in fixtures — metadata carries no
    # `finish_reason` key (the bridge passes response_metadata through
    # unchanged).
    assert "finish_reason" not in _event_metadata(finish)


def test_chunks_to_events_empty_iterator() -> None:
    """No chunks means no events."""
    assert list(chunks_to_events(iter([]))) == []


def test_chunks_to_events_block_transitions_keep_stable_indices() -> None:
    """String-keyed blocks that transition mid-stream each get their own lifecycle.

    Regression test for OpenAI `responses/v1` style streams where
    `content_blocks` uses string identifiers (e.g. `"lc_rs_305f30"`) to
    distinguish blocks. Each distinct block must get its own
    `content-block-start` / `content-block-finish` pair, with sequential
    `uint` wire indices, and deltas keep that stable wire index.
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

    starts: list[Any] = [e for e in events if e["event"] == "content-block-start"]
    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    assert [s["content"]["type"] for s in starts] == [
        "reasoning",
        "reasoning",
        "text",
    ]
    assert [f["content"]["type"] for f in finishes] == [
        "reasoning",
        "reasoning",
        "text",
    ]
    # Wire indices are sequential uints regardless of source-side keys.
    assert [s["index"] for s in starts] == [0, 1, 2]
    assert [f["index"] for f in finishes] == [0, 1, 2]

    # Blocks are finalized at message end so providers can interleave
    # deltas for parallel content blocks without closing them early.
    events_any: list[Any] = events
    lifecycle = [
        (e["event"], e["index"])
        for e in events_any
        if e["event"] in ("content-block-start", "content-block-finish")
    ]
    assert lifecycle == [
        ("content-block-start", 0),
        ("content-block-start", 1),
        ("content-block-start", 2),
        ("content-block-finish", 0),
        ("content-block-finish", 1),
        ("content-block-finish", 2),
    ]

    # Each finish carries the accumulated content for its block.
    assert finishes[0]["content"]["reasoning"] == "hmm then"
    assert finishes[1]["content"]["reasoning"] == "different"
    assert finishes[2]["content"]["text"] == "answer: 42"


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
    finish_events: list[Any] = [
        e for e in events if e["event"] == "content-block-finish"
    ]
    assert len(finish_events) == 1
    finalized = cast("ToolCall", finish_events[0]["content"])
    assert finalized["type"] == "tool_call"
    assert finalized["args"] == {"q": "test"}

    # No provider finish_reason in the fixture chunks — the bridge does
    # not synthesize one. It deliberately does not infer `"tool_use"`
    # from the presence of a valid tool_call either; terminal reasons
    # are provider-specific (see `_build_message_finish`).
    assert "finish_reason" not in _event_metadata(events[-1])


def test_chunks_to_events_interleaved_parallel_tool_calls() -> None:
    """Parallel tool-call chunks can interleave without losing block lifecycles."""
    events = list(
        chunks_to_events(
            iter(_interleaved_parallel_tool_call_chunks()), message_id="msg-1"
        )
    )

    _assert_interleaved_parallel_tool_call_events(events)


@pytest.mark.asyncio
async def test_achunks_to_events_interleaved_parallel_tool_calls() -> None:
    """Async bridge preserves lifecycles for interleaved parallel tool calls."""
    events = [
        event
        async for event in achunks_to_events(
            _aiter_chunks(_interleaved_parallel_tool_call_chunks()),
            message_id="msg-1",
        )
    ]

    _assert_interleaved_parallel_tool_call_events(events)


def _interleaved_parallel_tool_call_chunks() -> list[ChatGenerationChunk]:
    return [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id="msg-1",
                tool_call_chunks=[
                    {
                        "index": 0,
                        "id": "tc1",
                        "name": "task",
                        "args": '{"subagent_type": "haiku"',
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
                        "index": 1,
                        "id": "tc2",
                        "name": "task",
                        "args": '{"subagent_type": "limerick"',
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
                        "args": ', "description": "Write a haiku"}',
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
                        "index": 1,
                        "id": None,
                        "name": None,
                        "args": ', "description": "Write a limerick"}',
                        "type": "tool_call_chunk",
                    }
                ],
            )
        ),
    ]


async def _aiter_chunks(
    chunks: list[ChatGenerationChunk],
) -> AsyncIterator[ChatGenerationChunk]:
    for chunk in chunks:
        yield chunk


def _assert_interleaved_parallel_tool_call_events(events: list[Any]) -> None:
    assert_valid_event_stream(events)

    starts: list[Any] = [e for e in events if e["event"] == "content-block-start"]
    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    assert [s["index"] for s in starts] == [0, 1]
    assert [f["index"] for f in finishes] == [0, 1]

    finalized = [cast("ToolCall", event["content"]) for event in finishes]
    assert finalized[0]["id"] == "tc1"
    assert finalized[0]["args"] == {
        "subagent_type": "haiku",
        "description": "Write a haiku",
    }
    assert finalized[1]["id"] == "tc2"
    assert finalized[1]["args"] == {
        "subagent_type": "limerick",
        "description": "Write a limerick",
    }


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

    finish_events: list[Any] = [
        e for e in events if e["event"] == "content-block-finish"
    ]
    assert len(finish_events) == 1
    assert finish_events[0]["content"]["type"] == "invalid_tool_call"
    assert "finish_reason" not in _event_metadata(events[-1])


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
    finish_blocks: list[Any] = [
        e["content"] for e in events if e["event"] == "content-block-finish"
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
    finish_events: list[Any] = [
        e for e in events if e["event"] == "content-block-finish"
    ]
    assert [e["content"]["type"] for e in finish_events] == ["text"]


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
    finish_blocks: list[Any] = [
        e["content"] for e in events if e["event"] == "content-block-finish"
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
    finish_blocks: list[Any] = [
        e["content"] for e in events if e["event"] == "content-block-finish"
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
    assert start["id"] == "msg-1"

    delta_event = cast("ContentBlockDeltaData", events[2])
    assert delta_event["delta"] == {"type": "text-delta", "text": "Hello world"}

    final = cast("MessageFinishData", events[-1])
    assert "finish_reason" not in _event_metadata(final)


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

    starts: list[Any] = [e for e in events if e["event"] == "content-block-start"]
    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    assert [s["content"]["type"] for s in starts] == ["reasoning", "text"]
    assert [f["content"]["type"] for f in finishes] == ["reasoning", "text"]

    deltas: list[Any] = [e for e in events if e["event"] == "content-block-delta"]
    assert len(deltas) == 2
    assert deltas[0]["delta"] == {
        "type": "reasoning-delta",
        "reasoning": "think hard",
    }
    assert deltas[1]["delta"] == {"type": "text-delta", "text": "the answer"}


def test_message_to_events_tool_call_skips_delta() -> None:
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
    deltas: list[Any] = [e for e in events if e["event"] == "content-block-delta"]
    assert deltas == []

    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    assert len(finishes) == 1
    tc = cast("ToolCall", finishes[0]["content"])
    assert tc["type"] == "tool_call"
    assert tc["args"] == {"q": "hi"}

    # Message has no `finish_reason` / `stop_reason` in metadata; the
    # bridge does not synthesize one and does not second-guess based on
    # the presence of a tool_call.
    final = cast("MessageFinishData", events[-1])
    assert "finish_reason" not in _event_metadata(final)


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
    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    types = [f["content"]["type"] for f in finishes]
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

    # Passthrough: response_metadata lands on `metadata` unchanged,
    # including the raw provider `finish_reason`.
    final = cast("MessageFinishData", events[-1])
    assert _event_metadata(final) == {
        "finish_reason": "length",
        "model_name": "test-model",
        "stop_sequence": "</end>",
    }


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
    assert start["id"] == "msg-override"


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

    starts: list[Any] = [e for e in events if e["event"] == "content-block-start"]
    assert [s["content"]["type"] for s in starts] == [
        "image",
        "audio",
        "non_standard",
    ]

    image_start = starts[0]["content"]
    assert image_start["id"] == "img-1"
    assert image_start["mime_type"] == "image/png"
    assert "data" not in image_start

    audio_start = starts[1]["content"]
    assert audio_start["id"] == "aud-1"
    assert audio_start["mime_type"] == "audio/mp3"
    assert "data" not in audio_start
    assert "transcript" not in audio_start

    ns_start = starts[2]["content"]
    assert ns_start["type"] == "non_standard"
    assert ns_start["value"] == {}

    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    assert finishes[0]["content"]["data"] == "A" * 1024
    assert finishes[1]["content"]["data"] == "B" * 1024
    assert finishes[1]["content"]["transcript"] == "hello"
    assert finishes[2]["content"]["value"] == {"big": "C" * 1024}


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

    starts: list[Any] = [e for e in events if e["event"] == "content-block-start"]
    assert len(starts) == 1
    tc_start = starts[0]["content"]
    assert tc_start["type"] == "tool_call"
    assert tc_start["id"] == "tc1"
    assert tc_start["name"] == "search"
    assert tc_start["args"] == {}

    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    tc_finish = cast("ToolCall", finishes[0]["content"])
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


# ---------------------------------------------------------------------------
# Lifecycle validator: provider-style emission patterns
# ---------------------------------------------------------------------------


def _aimsg_chunk(blocks: list[CompatBlock], msg_id: str = "m") -> ChatGenerationChunk:
    """Wrap a list of content blocks into a ChatGenerationChunk.

    Matches what a provider's `_stream` would yield per SSE event.
    """
    return ChatGenerationChunk(message=AIMessageChunk(content=blocks, id=msg_id))


def test_lifecycle_validator_openai_chat_completions_style() -> None:
    """Text + streaming tool call with int indices, all at index 0/1.

    Mirrors OpenAI chat-completions API where each delta stays at the
    same integer index and a new tool call bumps the index.
    """
    chunks = [
        _aimsg_chunk([{"type": "text", "text": "Hello", "index": 0}]),
        _aimsg_chunk([{"type": "text", "text": " there", "index": 0}]),
    ]
    # Tool-call chunks go via the tool_call_chunks channel, not content.
    chunks.extend(
        [
            ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    id="m",
                    tool_call_chunks=[
                        {
                            "type": "tool_call_chunk",
                            "index": 1,
                            "id": "tc1",
                            "name": "lookup",
                            "args": '{"q":',
                        }
                    ],
                )
            ),
            ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    id="m",
                    tool_call_chunks=[
                        {
                            "type": "tool_call_chunk",
                            "index": 1,
                            "id": None,
                            "name": None,
                            "args": ' "pie"}',
                        }
                    ],
                )
            ),
        ]
    )

    events = list(chunks_to_events(iter(chunks), message_id="m"))
    assert_valid_event_stream(events)

    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    types = [f["content"]["type"] for f in finishes]
    assert types == ["text", "tool_call"]
    assert finishes[0]["content"]["text"] == "Hello there"
    assert finishes[1]["content"]["args"] == {"q": "pie"}


def test_lifecycle_validator_openai_responses_style() -> None:
    """Reasoning → text → reasoning → text with string block identifiers.

    Mirrors OpenAI `responses/v1` output_version where each distinct
    block has a string index like `lc_rs_305f30`.
    """
    chunks = [
        _aimsg_chunk([{"type": "reasoning", "reasoning": "hmm", "index": "rs_a"}]),
        _aimsg_chunk([{"type": "reasoning", "reasoning": " first", "index": "rs_a"}]),
        _aimsg_chunk([{"type": "text", "text": "Answer: ", "index": "txt_a"}]),
        _aimsg_chunk([{"type": "text", "text": "42", "index": "txt_a"}]),
        _aimsg_chunk([{"type": "reasoning", "reasoning": "actually", "index": "rs_b"}]),
        _aimsg_chunk([{"type": "text", "text": "42!", "index": "txt_b"}]),
    ]

    events = list(chunks_to_events(iter(chunks), message_id="m"))
    assert_valid_event_stream(events)

    starts: list[Any] = [e for e in events if e["event"] == "content-block-start"]
    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    # Four distinct blocks: reasoning, text, reasoning, text
    assert [s["content"]["type"] for s in starts] == [
        "reasoning",
        "text",
        "reasoning",
        "text",
    ]
    assert [s["index"] for s in starts] == [0, 1, 2, 3]
    assert [f["index"] for f in finishes] == [0, 1, 2, 3]
    assert finishes[0]["content"]["reasoning"] == "hmm first"
    assert finishes[1]["content"]["text"] == "Answer: 42"
    assert finishes[2]["content"]["reasoning"] == "actually"
    assert finishes[3]["content"]["text"] == "42!"


def test_lifecycle_validator_anthropic_style_text_and_thinking() -> None:
    """Interleaved text and thinking blocks with int indices.

    Mirrors Anthropic's per-event structure: one block per chunk, each
    labeled with an int `index` from Anthropic's content_block_start /
    content_block_delta events.
    """
    chunks = [
        _aimsg_chunk([{"type": "reasoning", "reasoning": "Let me think", "index": 0}]),
        _aimsg_chunk([{"type": "reasoning", "reasoning": " more", "index": 0}]),
        _aimsg_chunk([{"type": "text", "text": "The answer is", "index": 1}]),
        _aimsg_chunk([{"type": "text", "text": " 42.", "index": 1}]),
    ]

    events = list(chunks_to_events(iter(chunks), message_id="m"))
    assert_valid_event_stream(events)

    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == ["reasoning", "text"]
    assert finishes[0]["content"]["reasoning"] == "Let me think more"
    assert finishes[1]["content"]["text"] == "The answer is 42."


def test_lifecycle_validator_anthropic_reasoning_preserves_signature() -> None:
    """A later reasoning delta's `extras.signature` must land on the finish block.

    Anthropic emits reasoning content as `thinking_delta` events (text),
    followed by a `signature_delta` event carrying the cryptographic
    signature that the API requires on any follow-up turn. After the
    content-block-start/delta translation, that signature arrives as
    `extras.signature` on a reasoning delta that has no new text. If
    the bridge drops it, Claude rejects the next request with
    `messages.<n>.content.<k>.thinking.signature: Field required`.
    """
    chunks = [
        _aimsg_chunk([{"type": "reasoning", "reasoning": "Let me think", "index": 0}]),
        _aimsg_chunk([{"type": "reasoning", "reasoning": " more", "index": 0}]),
        # signature_delta arrives after the text; no new reasoning text
        # but carries the signature under `extras`.
        _aimsg_chunk(
            [
                {
                    "type": "reasoning",
                    "reasoning": "",
                    "index": 0,
                    "extras": {"signature": "sig-abc123"},
                }
            ]
        ),
        _aimsg_chunk([{"type": "text", "text": "Hi.", "index": 1}]),
    ]

    events = list(chunks_to_events(iter(chunks), message_id="m"))
    assert_valid_event_stream(events)

    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    reasoning_finish = finishes[0]["content"]
    assert reasoning_finish["type"] == "reasoning"
    assert reasoning_finish["reasoning"] == "Let me think more"
    assert reasoning_finish.get("extras", {}).get("signature") == "sig-abc123"


def test_lifecycle_validator_anthropic_style_tool_use_after_text() -> None:
    """Text then tool_use (tool_call_chunk) — Anthropic tool-calling pattern."""
    chunks = [
        _aimsg_chunk([{"type": "text", "text": "Looking up...", "index": 0}]),
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[],
                id="m",
                tool_call_chunks=[
                    {
                        "type": "tool_call_chunk",
                        "index": 1,
                        "id": "toolu_1",
                        "name": "search",
                        "args": "",
                    }
                ],
            )
        ),
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[],
                id="m",
                tool_call_chunks=[
                    {
                        "type": "tool_call_chunk",
                        "index": 1,
                        "id": None,
                        "name": None,
                        "args": '{"query": "42"}',
                    }
                ],
            )
        ),
    ]

    events = list(chunks_to_events(iter(chunks), message_id="m"))
    assert_valid_event_stream(events)

    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == ["text", "tool_call"]
    assert finishes[1]["content"]["args"] == {"query": "42"}
    assert finishes[1]["content"]["id"] == "toolu_1"


def test_lifecycle_validator_inline_image_block() -> None:
    """A self-contained image block gets start + finish with no delta."""
    chunks = [
        _aimsg_chunk(
            [
                {
                    "type": "image",
                    "id": "img1",
                    "mime_type": "image/png",
                    "data": "AAAA",
                    "index": 0,
                }
            ]
        ),
    ]
    events = list(chunks_to_events(iter(chunks), message_id="m"))
    assert_valid_event_stream(events)

    starts: list[Any] = [e for e in events if e["event"] == "content-block-start"]
    deltas: list[Any] = [e for e in events if e["event"] == "content-block-delta"]
    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    assert [s["content"]["type"] for s in starts] == ["image"]
    # Data payload rides as an explicit data-delta; start has heavy fields stripped.
    assert deltas == [
        {
            "event": "content-block-delta",
            "index": 0,
            "delta": {"type": "data-delta", "data": "AAAA"},
        }
    ]
    assert "data" not in starts[0]["content"]
    assert finishes[0]["content"]["data"] == "AAAA"


def test_lifecycle_validator_invalid_tool_call_args() -> None:
    """Malformed JSON args finalize to invalid_tool_call; lifecycle still valid."""
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id="m",
                tool_call_chunks=[
                    {
                        "type": "tool_call_chunk",
                        "index": 0,
                        "id": "bad1",
                        "name": "noop",
                        "args": "not json",
                    }
                ],
            )
        ),
    ]
    events = list(chunks_to_events(iter(chunks), message_id="m"))
    assert_valid_event_stream(events)

    finishes: list[Any] = [e for e in events if e["event"] == "content-block-finish"]
    assert len(finishes) == 1
    assert finishes[0]["content"]["type"] == "invalid_tool_call"


def test_lifecycle_validator_empty_stream() -> None:
    """An empty chunk iterator produces no events (and still validates)."""
    assert_valid_event_stream(list(chunks_to_events(iter([]))))


def test_lifecycle_validator_message_to_events_roundtrip() -> None:
    """`message_to_events` also produces spec-conformant lifecycles."""
    msg = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "think"},
            {"type": "text", "text": "answer"},
            {
                "type": "image",
                "id": "img1",
                "mime_type": "image/png",
                "data": "X" * 256,
            },
        ],
        id="msg-1",
        tool_calls=[
            {"id": "t1", "name": "search", "args": {"q": "pie"}, "type": "tool_call"},
        ],
    )
    events = list(message_to_events(msg))
    assert_valid_event_stream(events)
