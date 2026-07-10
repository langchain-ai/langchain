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


def test_tool_error_on_error_formats_content() -> None:
    """`on_error` sets the ToolMessage content and can use the tool call context."""

    def on_error(exc: Exception, request: ToolCallRequest) -> str:
        return (
            f"`{request.tool_call['name']}` raised {type(exc).__name__} for "
            f"{request.tool_call['args']}; fix the arguments and retry."
        )

    agent = create_agent(
        model=_model(),
        tools=[failing_tool],
        middleware=[ToolErrorMiddleware(catch=(ValueError,), on_error=on_error)],
    )

    result = agent.invoke({"messages": [HumanMessage("go")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert tool_messages[0].content == (
        "`failing_tool` raised ValueError for {'value': 'x'}; fix the arguments and retry."
    )
    # Custom formatter controls disclosure — the raw exception message is not leaked.
    assert "secret detail" not in tool_messages[0].content


async def test_tool_error_async_on_error() -> None:
    """An async `on_error` is awaited under async execution."""

    async def on_error(exc: Exception, request: ToolCallRequest) -> str:
        return f"async handled `{request.tool_call['name']}`: {type(exc).__name__}"

    agent = create_agent(
        model=_model(),
        tools=[failing_tool],
        middleware=[ToolErrorMiddleware(catch=(ValueError,), on_error=on_error)],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("go")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert tool_messages[0].content == "async handled `failing_tool`: ValueError"
