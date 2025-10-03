"""Tests for middleware decorators: before_model, after_model, and modify_model_request."""

import pytest
from typing import Any
from typing_extensions import NotRequired
from syrupy.assertion import SnapshotAssertion

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from langgraph.types import Command

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    before_model,
    after_model,
    modify_model_request,
    hook_config,
)
from langchain.agents.factory import create_agent, _get_can_jump_to

from ..model import FakeToolCallingModel


class CustomState(AgentState):
    """Custom state schema for testing."""

    custom_field: NotRequired[str]


@tool
def test_tool(input: str) -> str:
    """A test tool for middleware testing."""
    return f"Tool result: {input}"


def test_before_model_decorator() -> None:
    """Test before_model decorator with all configuration options."""

    @before_model(
        state_schema=CustomState, tools=[test_tool], can_jump_to=["end"], name="CustomBeforeModel"
    )
    def custom_before_model(state: CustomState, runtime: Runtime) -> dict[str, Any]:
        return {"jump_to": "end"}

    assert isinstance(custom_before_model, AgentMiddleware)
    assert custom_before_model.state_schema == CustomState
    assert custom_before_model.tools == [test_tool]
    assert getattr(custom_before_model.__class__.before_model, "__can_jump_to__", []) == ["end"]


def test_after_model_decorator() -> None:
    """Test after_model decorator with all configuration options."""

    @after_model(
        state_schema=CustomState, tools=[test_tool], can_jump_to=["end"], name="CustomAfterModel"
    )
    def custom_after_model(state: CustomState, runtime: Runtime) -> dict[str, Any]:
        return {"jump_to": "end"}

    assert isinstance(custom_after_model, AgentMiddleware)
    assert custom_after_model.state_schema == CustomState
    assert custom_after_model.tools == [test_tool]
    assert getattr(custom_after_model.__class__.after_model, "__can_jump_to__", []) == ["end"]


def test_modify_model_request_decorator() -> None:
    """Test modify_model_request decorator with all configuration options."""

    @modify_model_request(state_schema=CustomState, tools=[test_tool], name="CustomModifyRequest")
    def custom_modify_request(
        request: ModelRequest, state: CustomState, runtime: Runtime
    ) -> ModelRequest:
        return request

    assert isinstance(custom_modify_request, AgentMiddleware)
    assert custom_modify_request.state_schema == CustomState
    assert custom_modify_request.tools == [test_tool]


def test_decorator_with_minimal_config() -> None:
    """Test decorators with minimal configuration."""

    @before_model
    def minimal_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {}

    @after_model
    def minimal_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {}

    @modify_model_request
    def minimal_modify_request(
        request: ModelRequest, state: AgentState, runtime: Runtime
    ) -> ModelRequest:
        return request

    assert isinstance(minimal_before_model, AgentMiddleware)
    assert isinstance(minimal_after_model, AgentMiddleware)
    assert isinstance(minimal_modify_request, AgentMiddleware)


def test_decorator_with_jump_to() -> None:
    """Test decorators with can_jump_to configuration."""

    @before_model(can_jump_to=["end", "custom_node"])
    def jump_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {"jump_to": "end"}

    @after_model(can_jump_to=["end", "custom_node"])
    def jump_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {"jump_to": "end"}

    assert getattr(jump_before_model.__class__.before_model, "__can_jump_to__", []) == [
        "end",
        "custom_node",
    ]
    assert getattr(jump_after_model.__class__.after_model, "__can_jump_to__", []) == [
        "end",
        "custom_node",
    ]


def test_decorator_with_tools() -> None:
    """Test decorators with tools configuration."""

    @tool
    def tool1(input: str) -> str:
        """Tool 1."""
        return "result1"

    @tool
    def tool2(input: str) -> str:
        """Tool 2."""
        return "result2"

    @before_model(tools=[tool1, tool2])
    def tools_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {}

    assert tools_before_model.tools == [tool1, tool2]


def test_decorator_with_custom_state() -> None:
    """Test decorators with custom state schema."""

    class MyCustomState(AgentState):
        """Custom state for testing."""

        my_field: NotRequired[int]

    @before_model(state_schema=MyCustomState)
    def custom_state_before_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
        return {}

    assert custom_state_before_model.state_schema == MyCustomState


def test_decorator_with_name() -> None:
    """Test decorators with custom name."""

    @before_model(name="MyCustomMiddleware")
    def named_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {}

    assert named_before_model.__class__.__name__ == "MyCustomMiddleware"


def test_decorator_combined_config() -> None:
    """Test decorators with combined configuration options."""

    @tool
    def combined_tool(input: str) -> str:
        """Combined tool."""
        return "combined"

    class CombinedState(AgentState):
        """Combined state."""

        combined_field: NotRequired[str]

    @before_model(
        state_schema=CombinedState,
        tools=[combined_tool],
        can_jump_to=["end", "custom"],
        name="CombinedMiddleware",
    )
    def combined_before_model(state: CombinedState, runtime: Runtime) -> dict[str, Any]:
        return {"jump_to": "end"}

    assert isinstance(combined_before_model, AgentMiddleware)
    assert combined_before_model.state_schema == CombinedState
    assert combined_before_model.tools == [combined_tool]
    assert getattr(combined_before_model.__class__.before_model, "__can_jump_to__", []) == [
        "end",
        "custom",
    ]
    assert combined_before_model.__class__.__name__ == "CombinedMiddleware"


def test_get_can_jump_to() -> None:
    """Test _get_can_jump_to utility function."""

    @before_model(can_jump_to=["end", "custom"])
    def test_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {}

    can_jump_to = _get_can_jump_to(test_middleware)
    assert can_jump_to == ["end", "custom"]

    # Test with no can_jump_to
    @before_model
    def no_jump_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {}

    can_jump_to = _get_can_jump_to(no_jump_middleware)
    assert can_jump_to == []


def test_decorator_functionality() -> None:
    """Test that decorated functions work correctly in agents."""

    @before_model
    def test_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {"custom_field": "test_value"}

    @after_model
    def test_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {"processed": True}

    @modify_model_request
    def test_modify_request(
        request: ModelRequest, state: AgentState, runtime: Runtime
    ) -> ModelRequest:
        # Add custom header to model settings
        request.model_settings["custom_header"] = "test"
        return request

    # Create agent with decorated middleware
    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[test_tool],
        middleware=[test_before_model, test_after_model, test_modify_request],
    )

    # Test that agent can be invoked
    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert "messages" in result
    assert "custom_field" in result
    assert result["custom_field"] == "test_value"
    assert "processed" in result
    assert result["processed"] is True


def test_decorator_with_jump_functionality() -> None:
    """Test decorators with jump functionality."""

    @before_model(can_jump_to=["end"])
    def jump_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {"jump_to": "end"}

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[test_tool],
        middleware=[jump_middleware],
    )

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert "messages" in result
    # Should have jumped to end, so minimal messages
    assert len(result["messages"]) <= 2


def test_decorator_with_tools_in_agent() -> None:
    """Test that decorators with tools work correctly in agents."""

    @tool
    def decorator_tool(input: str) -> str:
        """Tool from decorator."""
        return f"Decorator tool result: {input}"

    @before_model(tools=[decorator_tool])
    def tools_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {}

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[test_tool],
        middleware=[tools_middleware],
    )

    # The agent should have access to both tools
    result = agent.invoke({"messages": [HumanMessage("Use both tools")]})
    assert "messages" in result


def test_decorator_error_handling() -> None:
    """Test decorator error handling."""

    @before_model
    def error_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        raise ValueError("Test error")

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[test_tool],
        middleware=[error_middleware],
    )

    # Should handle the error gracefully
    with pytest.raises(ValueError, match="Test error"):
        agent.invoke({"messages": [HumanMessage("Hello")]})


def test_decorator_with_hook_config() -> None:
    """Test decorators with hook_config."""

    @hook_config
    def my_hook_config():
        return {"custom_config": "test"}

    @before_model
    def config_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {}

    # Test that hook_config can be applied
    assert my_hook_config() == {"custom_config": "test"}


def test_decorator_snapshot_compatibility(snapshot: SnapshotAssertion) -> None:
    """Test that decorators produce consistent snapshots."""

    @before_model(name="SnapshotMiddleware")
    def snapshot_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        return {"snapshot_test": True}

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[test_tool],
        middleware=[snapshot_middleware],
    )

    result = agent.invoke({"messages": [HumanMessage("Snapshot test")]})
    assert result == snapshot
