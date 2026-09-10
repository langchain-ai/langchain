"""Tests for return_direct tool graph structure."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from syrupy.assertion import SnapshotAssertion

from langchain.agents.factory import create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel


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


def test_return_direct_tool_error_routes_back_to_model() -> None:
    """Test that a failed return_direct tool gives the model a chance to recover."""

    @tool(return_direct=True)
    def failing_tool(input_string: str) -> str:
        """A tool that always fails."""
        msg = f"failed to process {input_string}"
        raise ValueError(msg)

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {
                        "name": "failing_tool",
                        "args": {"input_string": "hello"},
                        "id": "call_1",
                    }
                ],
                [],
            ]
        ),
        tools=[failing_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Run the tool.")]})

    assert any(
        isinstance(message, ToolMessage)
        and message.tool_call_id == "call_1"
        and message.status == "error"
        for message in result["messages"]
    )
    assert isinstance(result["messages"][-1], AIMessage)
