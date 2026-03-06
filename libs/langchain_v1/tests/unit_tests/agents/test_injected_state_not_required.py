"""Tests for InjectedState with NotRequired state fields.

Verifies that tools annotated with ``InjectedState("<field>")`` do not
raise ``KeyError`` when the referenced field is declared as
``NotRequired`` in the state schema and has not been populated.

See: https://github.com/langchain-ai/langchain/issues/35585
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from typing_extensions import NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentState
from langchain.tools import InjectedState
from langchain.tools.tool_node import (
    ToolNode,
    _get_not_required_fields,
    _is_not_required,
)
from tests.unit_tests.agents.model import FakeToolCallingModel


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class _TotalTrueState(AgentState[Any]):
    """State where ``total=True`` (default) with a NotRequired field."""

    city: NotRequired[str]
    country: str


class _TotalFalseState(AgentState[Any], total=False):
    """State where ``total=False`` -- all fields are optional by default."""

    city: str
    country: str


def test_is_not_required_positive() -> None:
    """``_is_not_required`` returns True for NotRequired annotations."""
    assert _is_not_required(NotRequired[str]) is True
    assert _is_not_required(NotRequired[int]) is True


def test_is_not_required_negative() -> None:
    """``_is_not_required`` returns False for plain types."""
    assert _is_not_required(str) is False
    assert _is_not_required(int) is False
    assert _is_not_required(list[str]) is False


def test_get_not_required_fields_total_true() -> None:
    """Detect NotRequired fields in a total=True TypedDict."""
    fields = _get_not_required_fields(_TotalTrueState)
    assert "city" in fields
    # 'country' is required
    assert "country" not in fields


def test_get_not_required_fields_total_false() -> None:
    """In a total=False TypedDict all non-Required fields are optional."""
    fields = _get_not_required_fields(_TotalFalseState)
    assert "city" in fields
    assert "country" in fields


def test_get_not_required_fields_none_schema() -> None:
    """Passing None returns an empty set."""
    assert _get_not_required_fields(None) == set()


# ---------------------------------------------------------------------------
# Unit tests for ToolNode._is_field_not_required
# ---------------------------------------------------------------------------


def test_tool_node_is_field_not_required_with_schema() -> None:
    """ToolNode correctly identifies not-required fields from schema."""

    @tool
    def dummy_tool(x: int) -> str:
        """Dummy."""
        return str(x)

    node = ToolNode([dummy_tool], state_schema=_TotalTrueState)
    assert node._is_field_not_required("city") is True
    assert node._is_field_not_required("country") is False


def test_tool_node_is_field_not_required_without_schema() -> None:
    """Without schema all fields are treated as potentially not-required."""

    @tool
    def dummy_tool(x: int) -> str:
        """Dummy."""
        return str(x)

    node = ToolNode([dummy_tool])
    # No schema -> conservative: all fields use .get()
    assert node._is_field_not_required("anything") is True


# ---------------------------------------------------------------------------
# Integration tests with create_agent
# ---------------------------------------------------------------------------


def test_injected_state_not_required_field_absent() -> None:
    """InjectedState referencing a NotRequired field that is absent.

    This is the exact scenario reported in issue #35585.  Previously this
    raised ``KeyError: 'city'``.
    """

    class CustomState(AgentState[Any]):
        city: NotRequired[str]

    injected_values: dict[str, Any] = {}

    @tool
    def get_weather(city: Annotated[str, InjectedState("city")]) -> str:
        """Get weather for a given city."""
        injected_values["city"] = city
        if city is None:
            return "No city provided"
        return f"Sunny in {city}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {}, "id": "call_1", "name": "get_weather"}],
                [],
            ]
        ),
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
        state_schema=CustomState,
    )

    # Invoke WITHOUT providing the 'city' field -- previously crashed
    result = agent.invoke({"messages": [HumanMessage("What is the weather?")]})

    # The tool should have received None for the absent NotRequired field
    assert injected_values["city"] is None

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "No city provided" in tool_messages[0].content


def test_injected_state_not_required_field_present() -> None:
    """InjectedState referencing a NotRequired field that IS present.

    When the field is populated it should be injected normally.
    """

    class CustomState(AgentState[Any]):
        city: NotRequired[str]

    injected_values: dict[str, Any] = {}

    @tool
    def get_weather(city: Annotated[str, InjectedState("city")]) -> str:
        """Get weather for a given city."""
        injected_values["city"] = city
        return f"Sunny in {city}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {}, "id": "call_1", "name": "get_weather"}],
                [],
            ]
        ),
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
        state_schema=CustomState,
    )

    result = agent.invoke({
        "messages": [HumanMessage("Weather?")],
        "city": "Paris",
    })

    assert injected_values["city"] == "Paris"
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Sunny in Paris" in tool_messages[0].content


def test_injected_state_required_field_still_raises() -> None:
    """InjectedState referencing a required field should still KeyError.

    We only relax access for NotRequired fields; required fields that are
    missing should still surface an error so developers catch schema
    violations early.
    """

    class CustomState(AgentState[Any]):
        city: str  # required

    @tool
    def get_weather(city: Annotated[str, InjectedState("city")]) -> str:
        """Get weather for a given city."""
        return f"Sunny in {city}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {}, "id": "call_1", "name": "get_weather"}],
                [],
            ]
        ),
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
        state_schema=CustomState,
    )

    import pytest

    with pytest.raises(KeyError):
        agent.invoke({"messages": [HumanMessage("Weather?")]})


def test_injected_state_full_state_still_works() -> None:
    """InjectedState() (no field) should inject the full state dict."""

    class CustomState(AgentState[Any]):
        city: NotRequired[str]

    injected_values: dict[str, Any] = {}

    @tool
    def inspect_state(state: Annotated[dict, InjectedState()]) -> str:
        """Inspect full state."""
        injected_values["state"] = state
        return f"Keys: {sorted(state.keys())}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {}, "id": "call_1", "name": "inspect_state"}],
                [],
            ]
        ),
        tools=[inspect_state],
        system_prompt="You are a helpful assistant.",
        state_schema=CustomState,
    )

    result = agent.invoke({"messages": [HumanMessage("Inspect")]})

    assert "messages" in injected_values["state"]
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_injected_state_multiple_not_required_fields() -> None:
    """Multiple NotRequired fields, some absent and some present."""

    class CustomState(AgentState[Any]):
        city: NotRequired[str]
        temperature_unit: NotRequired[str]
        country: str

    injected_values: dict[str, Any] = {}

    @tool
    def weather_report(
        city: Annotated[str, InjectedState("city")],
        unit: Annotated[str, InjectedState("temperature_unit")],
        country: Annotated[str, InjectedState("country")],
    ) -> str:
        """Get weather report."""
        injected_values["city"] = city
        injected_values["unit"] = unit
        injected_values["country"] = country
        city_str = city or "unknown"
        unit_str = unit or "celsius"
        return f"Weather in {city_str}, {country}: 20 {unit_str}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {}, "id": "call_1", "name": "weather_report"}],
                [],
            ]
        ),
        tools=[weather_report],
        system_prompt="You are a helpful assistant.",
        state_schema=CustomState,
    )

    result = agent.invoke({
        "messages": [HumanMessage("Report")],
        "country": "France",
        # city and temperature_unit intentionally omitted
    })

    assert injected_values["city"] is None
    assert injected_values["unit"] is None
    assert injected_values["country"] == "France"

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "France" in tool_messages[0].content
