"""Tests for return_direct tool graph structure."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import ToolException
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
    """Test that a failed return_direct tool routes back to the model instead of exiting.

    When a return_direct=True tool raises an exception, ToolNode produces a
    ToolMessage with status="error". The agent should route back to the model
    so it can handle the error, rather than terminating immediately.
    """

    @tool(return_direct=True)
    def failing_tool(x: str) -> str:
        """A tool that always fails."""
        msg = "tool error"
        raise ToolException(msg)

    failing_tool.handle_tool_error = True

    tool_call = {"name": "failing_tool", "args": {"x": "hi"}, "id": "c1", "type": "tool_call"}
    model = FakeToolCallingModel(tool_calls=[[tool_call], []])
    agent = create_agent(
        model=model,
        tools=[failing_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("run the tool")]})

    # The agent must NOT terminate after the error ToolMessage.
    # It should route back to the model, which then produces a final AIMessage.
    last_message = result["messages"][-1]
    assert isinstance(last_message, AIMessage), (
        f"Expected final AIMessage after tool error, got {type(last_message).__name__}"
    )

    # Confirm the error ToolMessage is present in history
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"


def test_return_direct_tool_success_exits_loop() -> None:
    """Test that a successful return_direct tool exits the agent loop normally."""

    @tool(return_direct=True)
    def succeeding_tool(x: str) -> str:
        """A tool that always succeeds."""
        return f"result: {x}"

    tool_call = {"name": "succeeding_tool", "args": {"x": "hi"}, "id": "c2", "type": "tool_call"}
    model = FakeToolCallingModel(tool_calls=[[tool_call]])
    agent = create_agent(
        model=model,
        tools=[succeeding_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("run the tool")]})

    # Successful return_direct tool should exit immediately — last message is ToolMessage
    last_message = result["messages"][-1]
    assert isinstance(last_message, ToolMessage), (
        f"Expected ToolMessage as final message for successful return_direct, "
        f"got {type(last_message).__name__}"
    )
    assert last_message.status == "success"
