"""Unit tests for the Anthropic native stream-events converter."""

from typing import Any, cast

from anthropic.types import (
    InputJSONDelta,
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta as RawMessageDelta
from anthropic.types.raw_message_delta_event import (
    MessageDeltaUsage as RawMessageDeltaUsage,
)
from langchain_tests.utils.stream_lifecycle import assert_valid_event_stream

from langchain_anthropic import ChatAnthropic
from langchain_anthropic._stream_events import convert_anthropic_stream

MODEL_NAME = "claude-haiku-4-5-20251001"


def _events() -> list[Any]:
    msg = Message(
        id="msg_1",
        content=[],
        model=MODEL_NAME,
        role="assistant",
        stop_reason=None,
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=0),
        type="message",
    )
    return [
        RawMessageStartEvent(message=msg, type="message_start"),
        RawContentBlockStartEvent(
            content_block=ThinkingBlock(signature="", thinking="", type="thinking"),
            index=0,
            type="content_block_start",
        ),
        RawContentBlockDeltaEvent(
            delta=ThinkingDelta(thinking="Let me ", type="thinking_delta"),
            index=0,
            type="content_block_delta",
        ),
        RawContentBlockDeltaEvent(
            delta=ThinkingDelta(thinking="think.", type="thinking_delta"),
            index=0,
            type="content_block_delta",
        ),
        RawContentBlockStopEvent(index=0, type="content_block_stop"),
        RawContentBlockStartEvent(
            content_block=TextBlock(text="", type="text"),
            index=1,
            type="content_block_start",
        ),
        RawContentBlockDeltaEvent(
            delta=TextDelta(text="The answer ", type="text_delta"),
            index=1,
            type="content_block_delta",
        ),
        RawContentBlockDeltaEvent(
            delta=TextDelta(text="is 42.", type="text_delta"),
            index=1,
            type="content_block_delta",
        ),
        RawContentBlockStopEvent(index=1, type="content_block_stop"),
        RawContentBlockStartEvent(
            content_block=ToolUseBlock(
                id="toolu_1", input={}, name="search", type="tool_use"
            ),
            index=2,
            type="content_block_start",
        ),
        RawContentBlockDeltaEvent(
            delta=InputJSONDelta(partial_json='{"q":', type="input_json_delta"),
            index=2,
            type="content_block_delta",
        ),
        RawContentBlockDeltaEvent(
            delta=InputJSONDelta(partial_json=' "weather"}', type="input_json_delta"),
            index=2,
            type="content_block_delta",
        ),
        RawContentBlockStopEvent(index=2, type="content_block_stop"),
        RawMessageDeltaEvent(
            delta=RawMessageDelta(stop_reason="tool_use", stop_sequence=None),
            type="message_delta",
            usage=RawMessageDeltaUsage(
                output_tokens=50,
                input_tokens=10,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        ),
        RawMessageStopEvent(type="message_stop"),
    ]


def test_convert_anthropic_stream_lifecycle() -> None:
    llm = ChatAnthropic(model=MODEL_NAME)
    events: list[Any] = list(
        convert_anthropic_stream(
            iter(_events()), llm._make_message_chunk_from_anthropic_event
        )
    )

    assert_valid_event_stream(events)

    assert events[0]["event"] == "message-start"
    # The provider message id (`msg_1`) is deliberately NOT used: on the v3 path
    # core seeds the stream with the LangChain run id, and an empty id here lets
    # that stand (matching the compat bridge). Only an explicit `message_id`
    # overrides it.
    assert events[0]["id"] == ""
    assert events[0]["metadata"]["provider"] == "anthropic"
    assert events[0]["metadata"]["model"] == MODEL_NAME

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == [
        "reasoning",
        "text",
        "tool_call",
    ]
    assert [f["index"] for f in finishes] == [0, 1, 2]
    reasoning = cast("dict[str, Any]", finishes[0]["content"])
    text = cast("dict[str, Any]", finishes[1]["content"])
    assert reasoning["reasoning"] == "Let me think."
    assert text["text"] == "The answer is 42."
    tool = cast("dict[str, Any]", finishes[2]["content"])
    assert tool["args"] == {"q": "weather"}
    assert tool["name"] == "search"

    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    assert message_finish["metadata"]["stop_reason"] == "tool_use"
    # content-block-finish for index 0 arrives before index 1 starts
    # (true stop boundaries, not all-at-end).
    first_finish = next(
        i for i, e in enumerate(events) if e["event"] == "content-block-finish"
    )
    first_idx1_start = next(
        i
        for i, e in enumerate(events)
        if e["event"] == "content-block-start" and e["index"] == 1
    )
    assert first_finish < first_idx1_start


async def test_aconvert_anthropic_stream_lifecycle() -> None:
    llm = ChatAnthropic(model=MODEL_NAME)

    async def _araw() -> Any:
        for event in _events():
            yield event

    from langchain_anthropic._stream_events import aconvert_anthropic_stream

    events: list[Any] = [
        e
        async for e in aconvert_anthropic_stream(
            _araw(), llm._make_message_chunk_from_anthropic_event
        )
    ]
    assert_valid_event_stream(events)
    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == [
        "reasoning",
        "text",
        "tool_call",
    ]
    assert events[-1]["metadata"]["stop_reason"] == "tool_use"
