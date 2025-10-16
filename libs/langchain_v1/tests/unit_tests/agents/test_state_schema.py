"""Test state_schema parameter in create_agent.

This module tests that the state_schema parameter allows users to extend
AgentState without needing to create custom middleware.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentState
from langchain.tools import ToolRuntime

from .model import FakeToolCallingModel


def test_state_schema_basic() -> None:
    """Test that state_schema parameter works with a simple extension of AgentState."""

    class CustomState(AgentState):
        custom_field: str

    captured_state = {}

    @tool
    def custom_state_tool(x: int) -> str:
        """Simple tool."""
        return f"Processed {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 42}, "id": "call_123", "name": "custom_state_tool"}],
                [],
            ]
        ),
        tools=[custom_state_tool],
        system_prompt="You are a helpful assistant.",
        state_schema=CustomState,
    )

    result = agent.invoke({"messages": [HumanMessage("Test")], "custom_field": "test_value"})

    # Verify the custom field was preserved in the state
    assert result["custom_field"] == "test_value"
    assert len(result["messages"]) == 4


def test_state_schema_with_tool_runtime() -> None:
    """Test that state_schema works with ToolRuntime to access custom fields."""

    class ExtendedState(AgentState):
        counter: int

    runtime_data = {}

    @tool
    def counter_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that accesses custom state field."""
        runtime_data["counter"] = runtime.state.get("counter", 0)
        return f"Counter is {runtime_data['counter']}, x is {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 10}, "id": "call_1", "name": "counter_tool"}],
                [],
            ]
        ),
        tools=[counter_tool],
        system_prompt="You are a helpful assistant.",
        state_schema=ExtendedState,
    )

    result = agent.invoke({"messages": [HumanMessage("Test")], "counter": 5})

    # Verify custom field was accessible via ToolRuntime
    assert runtime_data["counter"] == 5
    assert "Counter is 5" in result["messages"][2].content


def test_state_schema_multiple_custom_fields() -> None:
    """Test state_schema with multiple custom fields."""

    class MultiFieldState(AgentState):
        user_id: str
        session_id: str
        context: str

    @tool
    def simple_tool(x: int) -> str:
        """Simple tool."""
        return f"Result: {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 1}, "id": "call_1", "name": "simple_tool"}],
                [],
            ]
        ),
        tools=[simple_tool],
        state_schema=MultiFieldState,
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage("Test")],
            "user_id": "user_123",
            "session_id": "session_456",
            "context": "test_context",
        }
    )

    # Verify all custom fields are preserved
    assert result["user_id"] == "user_123"
    assert result["session_id"] == "session_456"
    assert result["context"] == "test_context"


def test_state_schema_with_middleware() -> None:
    """Test that state_schema works alongside middleware state schemas."""
    from typing import Any

    from langchain.agents.middleware.types import AgentMiddleware

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

    @tool
    def state_tool(x: int) -> str:
        """Simple tool."""
        return f"x={x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 5}, "id": "call_1", "name": "state_tool"}],
                [],
            ]
        ),
        tools=[state_tool],
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

    # Verify both state schemas were merged
    assert result["user_name"] == "Alice"
    assert result["middleware_data"] == "test_data"
    assert "test_data" in middleware_calls


def test_state_schema_none_uses_default() -> None:
    """Test that when state_schema is None, the default AgentState is used."""

    @tool
    def basic_tool(x: int) -> str:
        """Basic tool."""
        return f"x={x}"

    # Create agent without state_schema (should work as before)
    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 1}, "id": "call_1", "name": "basic_tool"}],
                [],
            ]
        ),
        tools=[basic_tool],
        state_schema=None,  # Explicitly pass None
    )

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Verify basic functionality works
    assert len(result["messages"]) == 4
    assert "x=1" in result["messages"][2].content


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
            tool_calls=[
                [{"args": {"x": 99}, "id": "async_call", "name": "async_tool"}],
                [],
            ]
        ),
        tools=[async_tool],
        state_schema=AsyncState,
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Test async")], "async_field": "async_value"}
    )

    # Verify custom field was preserved
    assert result["async_field"] == "async_value"
    assert "Async: 99" in result["messages"][2].content


def test_state_schema_preserves_required_fields() -> None:
    """Test that state_schema preserves all standard AgentState fields."""

    class ExtendedAgentState(AgentState):
        extra_data: str

    @tool
    def check_tool(x: int) -> str:
        """Tool that returns a value."""
        return f"Result: {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 1}, "id": "call_1", "name": "check_tool"}],
                [],
            ]
        ),
        tools=[check_tool],
        state_schema=ExtendedAgentState,
    )

    # Input should have messages (required) and extra_data
    result = agent.invoke({"messages": [HumanMessage("Test")], "extra_data": "extra_value"})

    # Output should have messages and extra_data preserved
    assert len(result["messages"]) == 4
    assert result["extra_data"] == "extra_value"
