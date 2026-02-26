"""Tests for create_agent_tool — wrapping agents as tools for nested loops."""

import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.tools import tool

from langchain.agents import AgentSession, AgentState, create_agent, create_agent_tool
from tests.unit_tests.agents.model import FakeToolCallingModel

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)


def test_create_agent_tool_basic() -> None:
    """An inner agent wrapped as a tool returns the inner agent's final response."""

    @tool
    def inner_tool(query: str) -> str:
        """Inner tool that echoes."""
        return f"inner: {query}"

    inner_agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"query": "test"}, "id": "t1", "name": "inner_tool"}],
                [],
            ]
        ),
        tools=[inner_tool],
    )

    agent_tool = create_agent_tool(
        inner_agent,
        name="sub_agent",
        description="A sub-agent for research.",
    )

    assert agent_tool.name == "sub_agent"
    assert agent_tool.description == "A sub-agent for research."

    result = agent_tool.invoke("hello")
    assert isinstance(result, str)
    assert len(result) > 0


def test_create_agent_tool_no_tools() -> None:
    """An inner agent without tools returns a direct response."""
    inner_agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
    )

    agent_tool = create_agent_tool(
        inner_agent,
        name="simple_agent",
        description="A simple agent.",
    )

    result = agent_tool.invoke("hello")
    assert isinstance(result, str)
    assert len(result) > 0


def test_create_agent_tool_nested_in_outer_agent() -> None:
    """Outer agent invokes a sub-agent tool, creating a loop-in-loop pattern."""

    @tool
    def inner_tool(query: str) -> str:
        """Inner tool."""
        return f"inner: {query}"

    inner_model = FakeToolCallingModel(
        tool_calls=[
            [{"args": {"query": "delegated"}, "id": "t1", "name": "inner_tool"}],
            [],
        ]
    )
    inner_agent = create_agent(model=inner_model, tools=[inner_tool])

    sub_agent_tool = create_agent_tool(
        inner_agent,
        name="researcher",
        description="Delegate research tasks.",
    )

    outer_model = FakeToolCallingModel(
        tool_calls=[
            [{"args": {"query": "find info"}, "id": "o1", "name": "researcher"}],
            [],
        ]
    )
    outer_agent = create_agent(model=outer_model, tools=[sub_agent_tool])

    result = outer_agent.invoke({"messages": [HumanMessage("research something")]})
    messages = result["messages"]

    assert len(messages) >= 3
    assert isinstance(messages[0], HumanMessage)


@pytest.mark.anyio
async def test_create_agent_tool_async() -> None:
    """Async invocation of an agent tool works correctly."""

    @tool
    def echo_tool(query: str) -> str:
        """Echo tool."""
        return f"echo: {query}"

    inner_agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"query": "async_test"}, "id": "t1", "name": "echo_tool"}],
                [],
            ]
        ),
        tools=[echo_tool],
    )

    agent_tool = create_agent_tool(
        inner_agent,
        name="async_sub",
        description="Async sub-agent.",
    )

    result = await agent_tool.ainvoke("hello async")
    assert isinstance(result, str)
    assert len(result) > 0


def test_create_agent_tool_thread_id_prefix() -> None:
    """Thread ID prefix is applied when set."""
    inner_agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
    )

    agent_tool = create_agent_tool(
        inner_agent,
        name="prefixed",
        description="Agent with prefix.",
        thread_id_prefix="test-prefix",
    )

    result = agent_tool.invoke("hello")
    assert isinstance(result, str)


def test_create_agent_tool_is_exported() -> None:
    """create_agent_tool is importable from the agents package."""
    from langchain.agents import create_agent_tool as imported

    assert imported is not None
    assert callable(imported)
