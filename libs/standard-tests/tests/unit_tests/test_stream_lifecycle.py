"""Tests for the content-block protocol stream validator."""

from typing import Any

import pytest

from langchain_tests.utils.stream_lifecycle import assert_valid_event_stream

_ARGS_PARTS = ['{"path": ', '"a.py", ', '"content": ', '"hi"}']
_ARGS = {"path": "a.py", "content": "hi"}


def _tool_call_events(
    deltas: list[dict[str, Any]],
    *,
    finish_args: dict[str, Any] | None = None,
    start_args: str = "",
) -> list[dict[str, Any]]:
    """One tool call at index 0, with the given deltas between start and finish."""
    return [
        {"event": "message-start", "role": "ai"},
        {
            "event": "content-block-start",
            "index": 0,
            "content": {
                "type": "tool_call_chunk",
                "id": "c1",
                "name": "write_file",
                "args": start_args,
            },
        },
        *[
            {"event": "content-block-delta", "index": 0, "delta": delta}
            for delta in deltas
        ],
        {
            "event": "content-block-finish",
            "index": 0,
            "content": {
                "type": "tool_call",
                "id": "c1",
                "name": "write_file",
                "args": _ARGS if finish_args is None else finish_args,
            },
        },
        {"event": "message-finish"},
    ]


def _slices(parts: list[str]) -> list[dict[str, Any]]:
    return [{"type": "tool-call-delta", "args": part} for part in parts]


def _snapshots(parts: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "type": "block-delta",
            "fields": {
                "type": "tool_call_chunk",
                "args": "".join(parts[: i + 1]),
            },
        }
        for i in range(len(parts))
    ]


def test_tool_call_delta_args_must_match_finish() -> None:
    events = _tool_call_events(
        _slices(_ARGS_PARTS), finish_args={"path": "b.py", "content": "hi"}
    )

    with pytest.raises(AssertionError, match="tool_call args mismatch"):
        assert_valid_event_stream(events)


@pytest.mark.parametrize(
    "deltas",
    [
        pytest.param(_slices(_ARGS_PARTS), id="slices"),
        pytest.param(_snapshots(_ARGS_PARTS), id="snapshots"),
        pytest.param(
            [*_snapshots(_ARGS_PARTS[:2]), *_slices(_ARGS_PARTS[2:])],
            id="snapshot then slices",
        ),
    ],
)
def test_tool_call_args_in_either_form_are_valid(deltas: list[dict[str, Any]]) -> None:
    assert_valid_event_stream(_tool_call_events(deltas))


@pytest.mark.parametrize(
    "deltas",
    [
        pytest.param(_slices(_ARGS_PARTS[:-1]), id="slices"),
        pytest.param(_snapshots(_ARGS_PARTS)[:-1], id="snapshots"),
    ],
)
def test_incomplete_tool_call_args_are_invalid(deltas: list[dict[str, Any]]) -> None:
    with pytest.raises(AssertionError, match="tool_call args mismatch"):
        assert_valid_event_stream(_tool_call_events(deltas))


def test_unknown_delta_variant_is_accepted() -> None:
    deltas = [{"type": "future-delta", "args": "ignored"}, *_slices(_ARGS_PARTS)]

    assert_valid_event_stream(_tool_call_events(deltas))


def test_tool_call_delta_appends_to_args_on_the_start() -> None:
    events = _tool_call_events(_slices(_ARGS_PARTS[1:]), start_args=_ARGS_PARTS[0])

    assert_valid_event_stream(events)


def test_tool_call_delta_on_a_text_block_is_invalid() -> None:
    events = [
        {"event": "message-start", "role": "ai"},
        {
            "event": "content-block-start",
            "index": 0,
            "content": {"type": "text", "text": ""},
        },
        {
            "event": "content-block-delta",
            "index": 0,
            "delta": {"type": "tool-call-delta", "args": "{}"},
        },
        {
            "event": "content-block-finish",
            "index": 0,
            "content": {"type": "text", "text": ""},
        },
        {"event": "message-finish"},
    ]

    with pytest.raises(AssertionError, match="tool-call-delta"):
        assert_valid_event_stream(events)
