"""Unit tests for default_tools mutation isolation in create_agent.

Verifies that middleware mutating request.tools does not contaminate
subsequent agent invocations.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, ToolCall
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    InputAgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    OutputAgentState,
)
from langgraph.graph.state import CompiledStateGraph
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def static_tool(x: str) -> str:
    """A simple static tool."""
    return f"static:{x}"


class ToolTrackingMiddleware(AgentMiddleware):
    """Middleware that appends to request.tools and tracks initial lengths.

    On each invocation, records the initial tool count before mutation,
    then appends a duplicate of the first tool (which is already registered).
    If default_tools leaks across invocations, the initial count on the
    second call will be larger than the first.
    """

    def __init__(self) -> None:
        super().__init__()
        self.initial_tool_counts: list[int] = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        self.initial_tool_counts.append(len(request.tools))
        # Mutate the tools list in-place — the bug pattern
        if request.tools:
            request.tools.append(request.tools[0])
        return handler(request)


def test_default_tools_mutation_does_not_leak_across_invocations() -> None:
    """Middleware mutating request.tools should not affect subsequent calls.

    When middleware appends to request.tools, the mutation must be isolated
    to that single invocation. The second invocation should start with the
    same original tool count, not the mutated count from the first run.
    """
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="static_tool", args={"x": "test"}, id="1")],
            [ToolCall(name="static_tool", args={"x": "test"}, id="2")],
            [],
        ]
    )

    middleware = ToolTrackingMiddleware()
    agent: CompiledStateGraph[
        AgentState[Any], None, InputAgentState, OutputAgentState[Any]
    ] = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )

    agent.invoke(
        {"messages": [HumanMessage("first")]},
        {"configurable": {"thread_id": "1"}},
    )

    agent.invoke(
        {"messages": [HumanMessage("second")]},
        {"configurable": {"thread_id": "2"}},
    )

    # Every model call should start with exactly 1 tool (the static_tool),
    # regardless of how many times the model was called (agent loop iterations).
    # If the default_tools list leaked across calls, some counts would be > 1.
    assert all(c == 1 for c in middleware.initial_tool_counts), (
        f"Expected every model call to start with 1 tool, "
        f"got {middleware.initial_tool_counts}. "
        f"The default_tools list is being mutated across invocations."
    )
    # Sanity check: the model was called (at least once per invocation)
    assert len(middleware.initial_tool_counts) >= 2
