"""Tests for return_direct tool graph structure."""

import asyncio
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from syrupy.assertion import SnapshotAssertion

from langchain.agents.factory import create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain.agents.middleware.types import InputAgentState


def test_agent_graph_without_return_direct_tools(snapshot: SnapshotAssertion) -> None:
    """Test that graph WITHOUT return_direct tools does NOT have edge from tools to end."""

    @tool
    def normal_tool(input_string: str) -> str:
        """A normal tool without return_direct."""
        return input_string

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[normal_tool],
        system_prompt="You are a helpful assistant.",
    )

    # The mermaid diagram should NOT include an edge from tools to __end__
    # when no tools have return_direct=True
    mermaid_diagram = agent.get_graph().draw_mermaid()
    assert mermaid_diagram == snapshot


@pytest.mark.parametrize("async_execution", [False, True])
@pytest.mark.parametrize("error_type", ["validation", "execution"])
@pytest.mark.parametrize("parallel_success", [False, True])
@pytest.mark.parametrize("retry", [False, True])
async def test_return_direct_tool_error_returns_to_model(
    *, async_execution: bool, error_type: str, parallel_success: bool, retry: bool
) -> None:
    """Handled errors must reach the model, which may recover or retry successfully."""

    @tool(return_direct=True)
    def direct_tool(x: str) -> str:
        """Return the input, or fail when asked."""
        if x == "fail":
            msg = "tool error"
            raise ToolException(msg)
        return x

    direct_tool.handle_tool_error = True
    first_calls = [
        {
            "name": "direct_tool",
            "args": {"x": 1 if error_type == "validation" else "fail"},
            "id": "failed",
            "type": "tool_call",
        }
    ]
    if parallel_success:
        first_calls.append(
            {
                "name": "direct_tool",
                "args": {"x": "parallel success"},
                "id": "parallel",
                "type": "tool_call",
            }
        )
    retry_calls = [
        {
            "name": "direct_tool",
            "args": {"x": "recovered"},
            "id": "retry",
            "type": "tool_call",
        }
    ]
    model = FakeToolCallingModel(tool_calls=[first_calls, retry_calls if retry else [], []])
    agent = create_agent(model, tools=[direct_tool])
    inputs: InputAgentState = {"messages": [HumanMessage("run the tool")]}
    result = (
        await agent.ainvoke(inputs)
        if async_execution
        else await asyncio.to_thread(lambda: agent.invoke(inputs))
    )
    messages = result["messages"]
    results_by_id = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}

    assert results_by_id["failed"].status == "error"
    if error_type == "execution":
        assert results_by_id["failed"].content == "tool error"
    if parallel_success:
        assert results_by_id["parallel"].status == "success"
    assert len([m for m in messages if isinstance(m, AIMessage)]) == 2
    if retry:
        # An error from an earlier turn must not prevent a successful direct return.
        assert isinstance(messages[-1], ToolMessage)
        assert messages[-1].tool_call_id == "retry"
        assert messages[-1].status == "success"
        assert messages[-1].content == "recovered"
    else:
        assert isinstance(messages[-1], AIMessage)
        assert not messages[-1].tool_calls
        assert results_by_id["failed"].content in messages[-1].content


@pytest.mark.parametrize("async_execution", [False, True])
async def test_successful_return_direct_tool_exits(*, async_execution: bool) -> None:
    """Successful direct returns should not invoke the model again."""

    @tool(return_direct=True)
    def direct_tool(x: str) -> str:
        """Return the input."""
        return x

    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "direct_tool", "args": {"x": "hi"}, "id": "c1", "type": "tool_call"}],
            [],
        ]
    )
    agent = create_agent(model, tools=[direct_tool])
    inputs: InputAgentState = {"messages": [HumanMessage("run the tool")]}
    result = (
        await agent.ainvoke(inputs)
        if async_execution
        else await asyncio.to_thread(lambda: agent.invoke(inputs))
    )

    assert len(result["messages"]) == 3
    assert isinstance(result["messages"][-1], ToolMessage)
    assert result["messages"][-1].status == "success"
    assert result["messages"][-1].content == "hi"


def test_agent_graph_with_return_direct_tool(snapshot: SnapshotAssertion) -> None:
    """Test that graph WITH return_direct tools has correct edge from tools to end."""

    @tool(return_direct=True)
    def return_direct_tool(input_string: str) -> str:
        """A tool with return_direct=True."""
        return input_string

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[return_direct_tool],
        system_prompt="You are a helpful assistant.",
    )

    # The mermaid diagram SHOULD include an edge from tools to __end__
    # when at least one tool has return_direct=True
    mermaid_diagram = agent.get_graph().draw_mermaid()
    assert mermaid_diagram == snapshot


def test_agent_graph_with_mixed_tools(snapshot: SnapshotAssertion) -> None:
    """Test that graph with mixed tools (some return_direct, some not) has correct edges."""

    @tool(return_direct=True)
    def return_direct_tool(input_string: str) -> str:
        """A tool with return_direct=True."""
        return input_string

    @tool
    def normal_tool(input_string: str) -> str:
        """A normal tool without return_direct."""
        return input_string

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[return_direct_tool, normal_tool],
        system_prompt="You are a helpful assistant.",
    )

    # The mermaid diagram SHOULD include an edge from tools to __end__
    # because at least one tool has return_direct=True
    mermaid_diagram = agent.get_graph().draw_mermaid()
    assert mermaid_diagram == snapshot
