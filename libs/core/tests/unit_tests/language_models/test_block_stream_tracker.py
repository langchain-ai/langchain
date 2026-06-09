"""Unit tests for the shared BlockStreamTracker."""

from typing import Any

from langchain_core.language_models.stream_events import BlockStreamTracker


def test_feed_text_emits_start_then_delta() -> None:
    tracker = BlockStreamTracker()
    events: list[Any] = list(
        tracker.feed(0, {"type": "text", "text": "Hi", "index": 0})
    )
    assert events[0]["event"] == "content-block-start"
    assert events[0]["index"] == 0
    assert events[0]["content"] == {"type": "text", "text": ""}
    assert events[1]["event"] == "content-block-delta"
    assert events[1]["index"] == 0
    assert events[1]["delta"] == {"type": "text-delta", "text": "Hi"}


def test_feed_allocates_sequential_wire_indices() -> None:
    tracker = BlockStreamTracker()
    list(tracker.feed("a", {"type": "text", "text": "x", "index": "a"}))
    second: list[Any] = list(
        tracker.feed("b", {"type": "text", "text": "y", "index": "b"})
    )
    assert second[0]["index"] == 1  # second distinct key -> wire index 1


def test_finish_block_finalizes_tool_call_at_boundary() -> None:
    tracker = BlockStreamTracker()
    list(
        tracker.feed(
            0,
            {
                "type": "tool_call_chunk",
                "id": "t1",
                "name": "f",
                "args": '{"a":',
                "index": 0,
            },
        )
    )
    list(tracker.feed(0, {"type": "tool_call_chunk", "args": "1}", "index": 0}))
    finished: list[Any] = list(tracker.finish_block(0))
    assert len(finished) == 1
    assert finished[0]["event"] == "content-block-finish"
    assert finished[0]["content"]["type"] == "tool_call"
    assert finished[0]["content"]["args"] == {"a": 1}
    # finished block is closed: finish_all must not re-emit it
    assert list(tracker.finish_all()) == []


def test_finish_block_unknown_key_is_noop() -> None:
    assert list(BlockStreamTracker().finish_block("nope")) == []


def test_finish_all_finalizes_open_blocks_in_wire_order() -> None:
    tracker = BlockStreamTracker()
    list(tracker.feed(0, {"type": "text", "text": "hello", "index": 0}))
    list(tracker.feed(1, {"type": "reasoning", "reasoning": "why", "index": 1}))
    finished: list[Any] = list(tracker.finish_all())
    assert [f["index"] for f in finished] == [0, 1]
    assert finished[0]["content"]["text"] == "hello"
    assert finished[1]["content"]["reasoning"] == "why"
