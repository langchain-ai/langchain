"""Tests for TypeSafe state and LangChain message normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_typesafe._state import serialize_state

if TYPE_CHECKING:
    from langchain_typesafe.types import State


def test_empty_array_and_sequence_are_supported() -> None:
    """Empty native arrays and sequences both serialize to an empty array."""
    assert serialize_state([]) == []
    assert serialize_state(()) == []


def test_mixed_message_and_json_array_is_serialized_recursively() -> None:
    """Messages can appear alongside ordinary JSON values in an array."""
    state = cast(
        "State",
        [HumanMessage("hello"), {"priority": 1}, "plain JSON"],
    )

    assert serialize_state(state) == [
        {"role": "user", "content": "hello"},
        {"priority": 1},
        "plain JSON",
    ]


def test_messages_can_be_nested_inside_json_objects() -> None:
    """Message sequences and individual messages serialize at any object depth."""
    state: State = {
        "ticket": {
            "messages": (
                SystemMessage("You are a support assistant."),
                HumanMessage("My integration is broken."),
            ),
            "draft": AIMessage("I can help troubleshoot it."),
        },
        "priority": 2,
    }

    assert serialize_state(state) == {
        "ticket": {
            "messages": [
                {"role": "system", "content": "You are a support assistant."},
                {"role": "user", "content": "My integration is broken."},
            ],
            "draft": {
                "role": "assistant",
                "content": "I can help troubleshoot it.",
            },
        },
        "priority": 2,
    }


def test_unsupported_nested_state_value_is_rejected() -> None:
    """Unsupported objects are rejected even when nested in otherwise valid JSON."""
    state = cast("Any", {"ticket": {"attachment": object()}})

    with pytest.raises(TypeError, match="Unsupported TypeSafe state value: object"):
        serialize_state(state)


@pytest.mark.parametrize(
    "state",
    [
        {1: "value"},
        {"nested": {1: "value"}},
    ],
)
def test_non_string_state_key_is_rejected(state: Any) -> None:
    """Object keys must remain strings at every state nesting level."""
    with pytest.raises(TypeError, match="object keys must be strings"):
        serialize_state(state)


def test_json_tuple_is_serialized_as_array() -> None:
    """Python sequences of JSON values become TypeSafe state arrays."""
    assert serialize_state(("one", 2, None)) == ["one", 2, None]


def test_tool_message_is_serialized_with_tool_context() -> None:
    """Tool messages retain their role and tool-call relationship."""
    state = serialize_state(
        ToolMessage("Search result", tool_call_id="call_1", name="search")
    )

    assert state == {
        "role": "tool",
        "name": "search",
        "tool_call_id": "call_1",
        "content": "Search result",
    }


@pytest.mark.parametrize("state", [None, True, 42, 3.14, b"", bytearray()])
def test_invalid_root_state_is_rejected(state: Any) -> None:
    """Root state must be a string, object, array, or LangChain message."""
    with pytest.raises(TypeError, match="TypeSafe state"):
        serialize_state(state)
