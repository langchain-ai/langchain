"""Tests for return_direct tool graph structure and routing behavior."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, ToolException, tool
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


def test_return_direct_tool_success_terminates() -> None:
    """When a return_direct=True tool succeeds, the agent should exit immediately."""

    @tool(return_direct=True)
    def my_tool(x: str) -> str:
        """Return input directly."""
        return f"ok: {x}"

    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "my_tool", "args": {"x": "hello"}, "id": "c1", "type": "tool_call"}],
        ]
    )
    agent = create_agent(model, tools=[my_tool], system_prompt="You are a helpful assistant.")
    result = agent.invoke({"messages": [HumanMessage("run it")]})

    # The last message must be the ToolMessage (agent exited after tool success)
    assert isinstance(result["messages"][-1], ToolMessage)
    assert result["messages"][-1].status == "success"
    # Model was only invoked once (no retry loop)
    assert model.index == 1


def test_return_direct_tool_error_routes_to_model() -> None:
    """When a return_direct=True tool fails, the agent should route back to the model.

    The error ToolMessage should be passed to the model so it can retry or
    handle the failure, instead of the agent terminating silently with an error.
    """

    def _failing_impl(x: str) -> str:
        msg = "something went wrong"
        raise ToolException(msg)

    failing_tool = StructuredTool.from_function(
        func=_failing_impl,
        name="failing_tool",
        description="A tool that always fails.",
        return_direct=True,
        handle_tool_error=True,
    )

    # Round 1: model calls the tool; round 2: model produces a final text reply
    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "failing_tool", "args": {"x": "hi"}, "id": "c1", "type": "tool_call"}],
            [],
        ]
    )
    agent = create_agent(
        model, tools=[failing_tool], system_prompt="You are a helpful assistant."
    )
    result = agent.invoke({"messages": [HumanMessage("run it")]})

    # The final message must be an AIMessage, not the error ToolMessage
    assert isinstance(result["messages"][-1], AIMessage)

    # The error ToolMessage should be present in the history with status="error"
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"

    # Model was invoked twice: once to call the tool, once after the error
    assert model.index == 2
