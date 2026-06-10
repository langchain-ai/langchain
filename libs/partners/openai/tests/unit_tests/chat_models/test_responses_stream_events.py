"""Unit tests for the OpenAI Responses API native stream-events converter."""

from typing import Any, cast

from langchain_tests.utils.stream_lifecycle import assert_valid_event_stream

from langchain_openai.chat_models._stream_events import (
    convert_openai_responses_stream,
)
from langchain_openai.chat_models.base import (
    _convert_responses_chunk_to_generation_chunk,
)

# The shared fixture used by the existing bridge parity test.
from tests.unit_tests.chat_models.test_responses_stream import responses_stream


def test_convert_openai_responses_reasoning_lifecycle() -> None:
    events: list[Any] = list(
        convert_openai_responses_stream(
            iter(responses_stream),
            _convert_responses_chunk_to_generation_chunk,
            output_version="v1",
        )
    )
    assert_valid_event_stream(events)

    # message-start must NOT carry the provider response id (consistency with
    # the bridge / the rule from Phases 1-3): empty id lets core's seeded run id
    # stand.
    assert events[0]["event"] == "message-start"
    assert events[0]["id"] == ""
    assert events[0]["metadata"]["provider"] == "openai"

    reasoning_finishes = [
        e
        for e in events
        if e["event"] == "content-block-finish" and e["content"]["type"] == "reasoning"
    ]
    assert len(reasoning_finishes) == 4
    assert [
        cast("dict[str, Any]", f["content"])["reasoning"] for f in reasoning_finishes
    ] == [
        "reasoning block one",
        "another reasoning block",
        "more reasoning",
        "still more reasoning",
    ]
    assert events[-1]["event"] == "message-finish"


def test_convert_openai_responses_true_boundaries() -> None:
    """A block finishes before the next block's content arrives (true boundary)."""
    events: list[Any] = list(
        convert_openai_responses_stream(
            iter(responses_stream),
            _convert_responses_chunk_to_generation_chunk,
            output_version="v1",
        )
    )
    # The first content-block-finish must precede the start of a later index.
    first_finish_idx = next(
        i for i, e in enumerate(events) if e["event"] == "content-block-finish"
    )
    later_start_idx = next(
        (
            i
            for i, e in enumerate(events)
            if e["event"] == "content-block-start"
            and e["index"] > events[first_finish_idx]["index"]
        ),
        None,
    )
    # If there is a higher-index block, its start comes after the prior finish.
    if later_start_idx is not None:
        assert first_finish_idx < later_start_idx


async def test_aconvert_openai_responses_reasoning_lifecycle() -> None:
    async def _araw() -> Any:
        for c in responses_stream:
            yield c

    from langchain_openai.chat_models._stream_events import (
        aconvert_openai_responses_stream,
    )

    events: list[Any] = [
        e
        async for e in aconvert_openai_responses_stream(
            _araw(),
            _convert_responses_chunk_to_generation_chunk,
            output_version="v1",
        )
    ]
    assert_valid_event_stream(events)
    reasoning_finishes = [
        e
        for e in events
        if e["event"] == "content-block-finish" and e["content"]["type"] == "reasoning"
    ]
    assert len(reasoning_finishes) == 4
    assert events[-1]["event"] == "message-finish"
