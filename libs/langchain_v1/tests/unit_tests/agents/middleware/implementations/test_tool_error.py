"""Tests for ToolErrorMiddleware functionality."""

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool

from langchain.agents.factory import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def failing_tool(value: str) -> str:
    """Tool that always fails."""
    msg = f"secret detail: {value}"
    raise ValueError(msg)


def _model() -> FakeToolCallingModel:
    return FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"value": "x"}, id="1")],
            [],
        ]
    )


def test_tool_error_caught_returns_tool_message() -> None:
    """A caught exception becomes an error ToolMessage; default omits the raw message."""
    agent = create_agent(
        model=_model(),
        tools=[failing_tool],
        middleware=[ToolErrorMiddleware(catch=(ValueError,))],
    )

    result = agent.invoke({"messages": [HumanMessage("go")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert tool_messages[0].name == "failing_tool"
    assert "ValueError" in tool_messages[0].content
    # Default formatter must not leak the raw exception message.
    assert "secret detail" not in tool_messages[0].content


def test_tool_error_uncaught_propagates() -> None:
    """An exception not listed in `catch` propagates out of the agent."""
    agent = create_agent(
        model=_model(),
        tools=[failing_tool],
        middleware=[ToolErrorMiddleware(catch=(KeyError,))],
    )

    with pytest.raises(ValueError, match="secret detail"):
        agent.invoke({"messages": [HumanMessage("go")]})
