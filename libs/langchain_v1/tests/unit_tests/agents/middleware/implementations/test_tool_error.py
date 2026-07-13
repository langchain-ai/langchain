"""Tests for ToolErrorMiddleware functionality."""

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

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


def test_tool_error_handled_returns_tool_message() -> None:
    """`on_error` content is returned as an error ToolMessage and controls disclosure."""

    def on_error(exc: Exception, request: ToolCallRequest) -> str | None:
        return f"`{request.tool_call['name']}` failed with {type(exc).__name__}."

    agent = create_agent(
        model=_model(),
        tools=[failing_tool],
        middleware=[ToolErrorMiddleware(on_error)],
    )

    result = agent.invoke({"messages": [HumanMessage("go")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert tool_messages[0].name == "failing_tool"
    assert tool_messages[0].content == "`failing_tool` failed with ValueError."
    # on_error controls disclosure — the raw exception message is not leaked.
    assert "secret detail" not in tool_messages[0].content


def test_tool_error_none_propagates() -> None:
    """Returning `None` from `on_error` lets the exception propagate."""

    def on_error(exc: Exception, _request: ToolCallRequest) -> str | None:
        if isinstance(exc, KeyError):
            return "handled"
        return None  # ValueError is not handled -> propagates

    agent = create_agent(
        model=_model(),
        tools=[failing_tool],
        middleware=[ToolErrorMiddleware(on_error)],
    )

    with pytest.raises(ValueError, match="secret detail"):
        agent.invoke({"messages": [HumanMessage("go")]})


async def test_tool_error_async_only() -> None:
    """`aon_error` alone (no `on_error`) handles errors on the async path."""

    async def aon_error(exc: Exception, request: ToolCallRequest) -> str | None:
        return f"async handled `{request.tool_call['name']}`: {type(exc).__name__}"

    agent = create_agent(
        model=_model(),
        tools=[failing_tool],
        middleware=[ToolErrorMiddleware(aon_error=aon_error)],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("go")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert tool_messages[0].content == "async handled `failing_tool`: ValueError"


def test_tool_error_requires_a_handler() -> None:
    """At least one of `on_error`/`aon_error` must be provided."""
    with pytest.raises(ValueError, match="on_error"):
        ToolErrorMiddleware()
