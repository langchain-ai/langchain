"""Validator for LangChain content-block protocol event streams.

Checks that an event stream emitted by a chat model (via `stream_v2`,
or by the compat bridge's `chunks_to_events` / `message_to_events`)
conforms to the protocol lifecycle rules:

- `message-start` opens and `message-finish` closes the stream.
- Content blocks do not interleave: each block runs
  `content-block-start` → optional `content-block-delta`s →
  `content-block-finish` before the next block begins.
- Wire indices on content-block events are sequential `uint` values
  starting at 0.
- For deltaable block types (`text`, `reasoning`, `tool_call_chunk`,
  `server_tool_call_chunk`), accumulated delta content matches the
  final payload delivered on `content-block-finish`.

The validator accepts any iterable of protocol event dicts. It raises
`AssertionError` on the first violation with a descriptive message.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


_DELTAABLE_TYPES = frozenset(
    {"text", "reasoning", "tool_call_chunk", "server_tool_call_chunk"}
)


def assert_valid_event_stream(events: Iterable[Any]) -> None:
    """Assert that a stream of protocol events obeys the lifecycle contract.

    Args:
        events: Iterable of protocol event dicts (as yielded by
            `stream_v2` or `chunks_to_events`).

    Raises:
        AssertionError: On the first lifecycle violation found. The
            message identifies the event index and the specific rule
            that was broken.
    """
    event_list = list(events)
    if not event_list:
        return

    first = event_list[0]
    assert first["event"] == "message-start", (
        f"first event must be `message-start`, got {first['event']!r}"
    )
    message_start_positions = [
        i for i, e in enumerate(event_list) if e["event"] == "message-start"
    ]
    assert message_start_positions == [0], (
        f"expected exactly one `message-start` at position 0, "
        f"got positions {message_start_positions}"
    )

    message_finish_positions = [
        i for i, e in enumerate(event_list) if e["event"] == "message-finish"
    ]
    assert len(message_finish_positions) <= 1, (
        f"expected at most one `message-finish`, got {len(message_finish_positions)}"
    )
    if message_finish_positions:
        assert message_finish_positions[0] == len(event_list) - 1, (
            "`message-finish` must be the final event"
        )

    open_idx: int | None = None
    expected_next_idx = 0
    start_events: dict[int, dict[str, Any]] = {}
    finish_events: dict[int, dict[str, Any]] = {}
    delta_accum: dict[int, dict[str, Any]] = {}

    for i, event in enumerate(event_list):
        ev = event["event"]
        if ev == "message-start":
            assert i == 0, f"duplicate `message-start` at event {i}"
            continue
        if ev == "message-finish":
            assert open_idx is None, (
                f"`message-finish` while block {open_idx} still open (event {i})"
            )
            continue
        if ev == "error":
            continue
        if ev == "content-block-start":
            idx = event["index"]
            assert isinstance(idx, int), (
                f"content-block-start wire index must be an int, "
                f"got {idx!r} at event {i}"
            )
            assert idx >= 0, (
                f"content-block-start wire index must be non-negative, "
                f"got {idx} at event {i}"
            )
            assert idx == expected_next_idx, (
                f"expected next wire index {expected_next_idx}, got {idx} at event {i}"
            )
            assert open_idx is None, (
                f"content-block-start at idx={idx} while block {open_idx} "
                f"still open (event {i}); blocks must not interleave"
            )
            open_idx = idx
            start_events[idx] = event["content_block"]
            delta_accum[idx] = {}
            expected_next_idx += 1
        elif ev == "content-block-delta":
            idx = event["index"]
            assert idx == open_idx, (
                f"content-block-delta at idx={idx} but currently-open block is "
                f"{open_idx} (event {i})"
            )
            block = event["content_block"]
            _accumulate_delta(delta_accum[idx], block)
        elif ev == "content-block-finish":
            idx = event["index"]
            assert idx == open_idx, (
                f"content-block-finish at idx={idx} but currently-open block is "
                f"{open_idx} (event {i})"
            )
            finish_events[idx] = event["content_block"]
            open_idx = None
        else:
            # Unknown event types are accepted; the CDDL allows extensions.
            continue

    assert open_idx is None, (
        f"block {open_idx} still open at end of stream — no content-block-finish"
    )
    missing = set(start_events) - set(finish_events)
    assert not missing, (
        f"the following block indices have no content-block-finish event: "
        f"{sorted(missing)}"
    )

    for idx, finish_block in finish_events.items():
        _assert_delta_matches_finish(idx, delta_accum[idx], finish_block)


def _accumulate_delta(accum: dict[str, Any], block: dict[str, Any]) -> None:
    """Fold a delta block into the running accumulator for its index."""
    btype = block.get("type")
    if btype not in _DELTAABLE_TYPES:
        return
    if btype == "text":
        accum["text"] = accum.get("text", "") + block.get("text", "")
    elif btype == "reasoning":
        accum["reasoning"] = accum.get("reasoning", "") + block.get("reasoning", "")
    else:  # tool_call_chunk / server_tool_call_chunk
        accum["args"] = accum.get("args", "") + (block.get("args") or "")
        if block.get("id") is not None:
            accum["id"] = block["id"]
        if block.get("name") is not None:
            accum["name"] = block["name"]


def _assert_delta_matches_finish(
    idx: int,
    accum: dict[str, Any],
    finish_block: dict[str, Any],
) -> None:
    """Assert accumulated delta content is reflected in the finish payload."""
    ftype = finish_block.get("type")
    if ftype == "text" and "text" in accum:
        assert finish_block.get("text", "") == accum["text"], (
            f"block {idx} text accumulation {accum['text']!r} does not match "
            f"finish text {finish_block.get('text', '')!r}"
        )
    elif ftype == "reasoning" and "reasoning" in accum:
        assert finish_block.get("reasoning", "") == accum["reasoning"], (
            f"block {idx} reasoning accumulation mismatch: "
            f"accumulated {accum['reasoning']!r}, finish "
            f"{finish_block.get('reasoning', '')!r}"
        )
    elif ftype == "tool_call" and "args" in accum:
        # tool_call_chunk args are concatenated partial-JSON strings that
        # parse to a dict on finish.
        try:
            parsed = json.loads(accum["args"]) if accum["args"] else {}
        except json.JSONDecodeError:
            # Finish upgrades malformed args to invalid_tool_call, not
            # tool_call — so a tool_call finish implies args parsed cleanly.
            parsed = None
        assert finish_block.get("args") == parsed, (
            f"block {idx} tool_call args mismatch: accumulated parse "
            f"{parsed!r}, finish {finish_block.get('args')!r}"
        )
    elif ftype == "server_tool_call" and "args" in accum:
        try:
            parsed = json.loads(accum["args"]) if accum["args"] else {}
        except json.JSONDecodeError:
            parsed = None
        assert finish_block.get("args") == parsed


__all__ = ["assert_valid_event_stream"]
