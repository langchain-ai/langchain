"""Tests for the langchain ToolNode subclass (NotRequired state field handling)."""

from __future__ import annotations

from typing import Annotated
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolRuntime
from typing_extensions import NotRequired, TypedDict

from langchain.tools.tool_node import ToolNode

# -- helpers ----------------------------------------------------------------


class StateWithOptional(TypedDict):
    messages: list[AIMessage]
    city: NotRequired[str]


@tool
def get_weather(city: Annotated[str, InjectedState("city")]) -> str:
    """Get weather for a given city."""
    return f"Sunny in {city}"


@tool
def get_full_state(state: Annotated[dict[str, object], InjectedState()]) -> str:
    """Tool that receives the full state."""
    return str(state)


# -- tests ------------------------------------------------------------------


def test_inject_state_field_present() -> None:
    """InjectedState works normally when the referenced field IS in state."""
    node = ToolNode(tools=[get_weather])
    tc: ToolCall = {
        "name": "get_weather",
        "args": {},
        "id": "call_1",
        "type": "tool_call",
    }
    runtime = MagicMock(spec=ToolRuntime)
    runtime.state = {"messages": [], "city": "Rome"}

    result = node._inject_tool_args(tc, runtime)
    assert result["args"]["city"] == "Rome"


def test_inject_state_not_required_field_absent() -> None:
    """InjectedState must not raise KeyError when a NotRequired field is absent.

    This is the core regression test for
    https://github.com/langchain-ai/langchain/issues/35585
    """
    node = ToolNode(tools=[get_weather])
    tc: ToolCall = {
        "name": "get_weather",
        "args": {},
        "id": "call_2",
        "type": "tool_call",
    }
    runtime = MagicMock(spec=ToolRuntime)
    runtime.state = {"messages": []}  # "city" is absent

    # Before the fix this raised KeyError: 'city'
    result = node._inject_tool_args(tc, runtime)
    assert result["args"]["city"] is None


def test_inject_full_state_when_field_is_none() -> None:
    """When InjectedState() has no field, the entire state dict is injected."""
    node = ToolNode(tools=[get_full_state])
    tc: ToolCall = {
        "name": "get_full_state",
        "args": {},
        "id": "call_3",
        "type": "tool_call",
    }
    state_dict = {"messages": [AIMessage(content="hi", tool_calls=[])]}
    runtime = MagicMock(spec=ToolRuntime)
    runtime.state = state_dict

    result = node._inject_tool_args(tc, runtime)
    assert result["args"]["state"] is state_dict


def test_inject_state_object_attr_missing() -> None:
    """Handles missing attributes on non-dict state objects gracefully."""

    class ObjState:
        def __init__(self) -> None:
            self.messages: list[AIMessage] = []

    node = ToolNode(tools=[get_weather])
    tc: ToolCall = {
        "name": "get_weather",
        "args": {},
        "id": "call_4",
        "type": "tool_call",
    }
    runtime = MagicMock(spec=ToolRuntime)
    runtime.state = ObjState()  # no 'city' attribute

    result = node._inject_tool_args(tc, runtime)
    assert result["args"]["city"] is None
