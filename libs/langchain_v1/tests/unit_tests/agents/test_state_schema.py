"""Test state_schema parameter in create_agent.

This module tests that the state_schema parameter allows users to extend
AgentState without needing to create custom middleware.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState

# Cannot move ToolRuntime to TYPE_CHECKING as parameters of @tool annotated functions
# are inspected at runtime.
from langchain.tools import ToolRuntime  # noqa: TC001
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def simple_tool(x: int) -> str:
    """Simple tool for basic tests."""
    return f"Result: {x}"


def test_state_schema_single_custom_field() -> None:
    """Test that a single custom state field is preserved through agent execution."""

    class CustomState(AgentState):
        custom_field: str

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 1}, "id": "call_1", "name": "simple_tool"}], []]
        ),
        tools=[simple_tool],
        state_schema=CustomState,
    )

    result = agent.invoke({"messages": [HumanMessage("Test")], "custom_field": "test_value"})

    assert result["custom_field"] == "test_value"
    assert len(result["messages"]) == 4


def test_state_schema_multiple_custom_fields() -> None:
    """Test that multiple custom state fields are preserved through agent execution."""

    class CustomState(AgentState):
        user_id: str
        session_id: str
        context: str

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 1}, "id": "call_1", "name": "simple_tool"}], []]
        ),
        tools=[simple_tool],
        state_schema=CustomState,
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage("Test")],
            "user_id": "user_123",
            "session_id": "session_456",
            "context": "test_ctx",
        }
    )

    assert result["user_id"] == "user_123"
    assert result["session_id"] == "session_456"
    assert result["context"] == "test_ctx"
    assert len(result["messages"]) == 4


def test_state_schema_with_tool_runtime() -> None:
    """Test that custom state fields are accessible via ToolRuntime."""

    class ExtendedState(AgentState):
        counter: int

    runtime_data = {}

    @tool
    def counter_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that accesses custom state field."""
        runtime_data["counter"] = runtime.state["counter"]
        return f"Counter is {runtime_data['counter']}, x is {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 10}, "id": "call_1", "name": "counter_tool"}], []]
        ),
        tools=[counter_tool],
        state_schema=ExtendedState,
    )

    result = agent.invoke({"messages": [HumanMessage("Test")], "counter": 5})

    assert runtime_data["counter"] == 5
    assert "Counter is 5" in result["messages"][2].content


def test_state_schema_with_middleware() -> None:
    """Test that state_schema merges with middleware state schemas."""

    class UserState(AgentState):
        user_name: str

    class MiddlewareState(AgentState):
        middleware_data: str

    middleware_calls = []

    class TestMiddleware(AgentMiddleware):
        state_schema = MiddlewareState

        def before_model(self, state, runtime) -> dict[str, Any]:
            middleware_calls.append(state.get("middleware_data", ""))
            return {}

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 5}, "id": "call_1", "name": "simple_tool"}], []]
        ),
        tools=[simple_tool],
        state_schema=UserState,
        middleware=[TestMiddleware()],
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage("Test")],
            "user_name": "Alice",
            "middleware_data": "test_data",
        }
    )

    assert result["user_name"] == "Alice"
    assert result["middleware_data"] == "test_data"
    assert "test_data" in middleware_calls


def test_state_schema_none_uses_default() -> None:
    """Test that state_schema=None uses default AgentState."""
    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 1}, "id": "call_1", "name": "simple_tool"}], []]
        ),
        tools=[simple_tool],
        state_schema=None,
    )

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    assert len(result["messages"]) == 4
    assert "Result: 1" in result["messages"][2].content


async def test_state_schema_async() -> None:
    """Test that state_schema works with async agents."""

    class AsyncState(AgentState):
        async_field: str

    @tool
    async def async_tool(x: int) -> str:
        """Async tool."""
        return f"Async: {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 99}, "id": "call_1", "name": "async_tool"}], []]
        ),
        tools=[async_tool],
        state_schema=AsyncState,
    )

    result = await agent.ainvoke(
        {
            "messages": [HumanMessage("Test async")],
            "async_field": "async_value",
        }
    )

    assert result["async_field"] == "async_value"
    assert "Async: 99" in result["messages"][2].content
