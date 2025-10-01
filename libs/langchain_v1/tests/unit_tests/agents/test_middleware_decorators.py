"""Consolidated tests for middleware decorators: before_model, after_model, and modify_model_request."""

import pytest
from typing import Any
from typing_extensions import NotRequired

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
)
from langchain.agents.middleware_agent import create_agent
from .model import FakeToolCallingModel


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
        state_schema=CustomState, tools=[test_tool], jump_to=["__end__"], name="CustomBeforeModel"
    )
    def custom_before_model(state: CustomState, runtime: Runtime) -> dict[str, Any]:
        return {"jump_to": "__end__"}

    assert isinstance(custom_before_model, AgentMiddleware)
    assert custom_before_model.state_schema == CustomState
    assert custom_before_model.tools == [test_tool]
    assert custom_before_model.before_model_jump_to == ["__end__"]
    assert custom_before_model.__class__.__name__ == "CustomBeforeModel"

    result = custom_before_model.before_model({"messages": [HumanMessage("Hello")]}, None)
    assert result == {"jump_to": "__end__"}


def test_after_model_decorator() -> None:
    """Test after_model decorator with all configuration options."""

    @after_model(
        state_schema=CustomState,
        tools=[test_tool],
        jump_to=["model", "__end__"],
        name="CustomAfterModel",
    )
    def custom_after_model(state: CustomState, runtime: Runtime) -> dict[str, Any]:
        return {"jump_to": "model"}

    # Verify all options were applied
    assert isinstance(custom_after_model, AgentMiddleware)
    assert custom_after_model.state_schema == CustomState
    assert custom_after_model.tools == [test_tool]
    assert custom_after_model.after_model_jump_to == ["model", "__end__"]
    assert custom_after_model.__class__.__name__ == "CustomAfterModel"

    # Verify it works
    result = custom_after_model.after_model(
        {"messages": [HumanMessage("Hello"), AIMessage("Hi!")]}, None
    )
    assert result == {"jump_to": "model"}


def test_modify_model_request_decorator() -> None:
    """Test modify_model_request decorator with all configuration options."""

    @modify_model_request(state_schema=CustomState, tools=[test_tool], name="CustomModifyRequest")
    def custom_modify_request(
        request: ModelRequest, state: CustomState, runtime: Runtime
    ) -> ModelRequest:
        request.system_prompt = "Modified"
        return request

    # Verify all options were applied
    assert isinstance(custom_modify_request, AgentMiddleware)
    assert custom_modify_request.state_schema == CustomState
    assert custom_modify_request.tools == [test_tool]
    assert custom_modify_request.__class__.__name__ == "CustomModifyRequest"

    # Verify it works
    original_request = ModelRequest(
        model="test-model",
        system_prompt="Original",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
    )
    result = custom_modify_request.modify_model_request(
        original_request, {"messages": [HumanMessage("Hello")]}, None
    )
    assert result.system_prompt == "Modified"


def test_all_decorators_integration() -> None:
    """Test all three decorators working together in an agent."""
    call_order = []

    @before_model
    def track_before(state: AgentState, runtime: Runtime) -> None:
        call_order.append("before")
        return None

    @modify_model_request
    def track_modify(request: ModelRequest, state: AgentState, runtime: Runtime) -> ModelRequest:
        call_order.append("modify")
        return request

    @after_model
    def track_after(state: AgentState, runtime: Runtime) -> None:
        call_order.append("after")
        return None

    agent = create_agent(
        model=FakeToolCallingModel(), middleware=[track_before, track_modify, track_after]
    )
    agent = agent.compile()
    agent.invoke({"messages": [HumanMessage("Hello")]})

    assert call_order == ["before", "modify", "after"]


def test_decorators_use_function_names_as_default() -> None:
    """Test that decorators use function names as default middleware names."""

    @before_model
    def my_before_hook(state: AgentState, runtime: Runtime) -> None:
        return None

    @modify_model_request
    def my_modify_hook(request: ModelRequest, state: AgentState, runtime: Runtime) -> ModelRequest:
        return request

    @after_model
    def my_after_hook(state: AgentState, runtime: Runtime) -> None:
        return None

    # Verify that function names are used as middleware class names
    assert my_before_hook.__class__.__name__ == "my_before_hook"
    assert my_modify_hook.__class__.__name__ == "my_modify_hook"
    assert my_after_hook.__class__.__name__ == "my_after_hook"


# Async Decorator Tests


def test_async_before_model_decorator() -> None:
    """Test before_model decorator with async function."""

    @before_model(state_schema=CustomState, tools=[test_tool], name="AsyncBeforeModel")
    async def async_before_model(state: CustomState, runtime: Runtime) -> dict[str, Any]:
        return {"custom_field": "async_value"}

    assert isinstance(async_before_model, AgentMiddleware)
    assert async_before_model.state_schema == CustomState
    assert async_before_model.tools == [test_tool]
    assert async_before_model.__class__.__name__ == "AsyncBeforeModel"


def test_async_after_model_decorator() -> None:
    """Test after_model decorator with async function."""

    @after_model(state_schema=CustomState, tools=[test_tool], name="AsyncAfterModel")
    async def async_after_model(state: CustomState, runtime: Runtime) -> dict[str, Any]:
        return {"custom_field": "async_value"}

    assert isinstance(async_after_model, AgentMiddleware)
    assert async_after_model.state_schema == CustomState
    assert async_after_model.tools == [test_tool]
    assert async_after_model.__class__.__name__ == "AsyncAfterModel"


def test_async_modify_model_request_decorator() -> None:
    """Test modify_model_request decorator with async function."""

    @modify_model_request(state_schema=CustomState, tools=[test_tool], name="AsyncModifyRequest")
    async def async_modify_request(
        request: ModelRequest, state: CustomState, runtime: Runtime
    ) -> ModelRequest:
        request.system_prompt = "Modified async"
        return request

    assert isinstance(async_modify_request, AgentMiddleware)
    assert async_modify_request.state_schema == CustomState
    assert async_modify_request.tools == [test_tool]
    assert async_modify_request.__class__.__name__ == "AsyncModifyRequest"


def test_mixed_sync_async_decorators() -> None:
    """Test decorators with both sync and async functions."""

    @before_model(name="MixedBeforeModel")
    def sync_before(state: AgentState, runtime: Runtime) -> None:
        return None

    @before_model(name="MixedBeforeModel")
    async def async_before(state: AgentState, runtime: Runtime) -> None:
        return None

    @modify_model_request(name="MixedModifyRequest")
    def sync_modify(request: ModelRequest, state: AgentState, runtime: Runtime) -> ModelRequest:
        return request

    @modify_model_request(name="MixedModifyRequest")
    async def async_modify(
        request: ModelRequest, state: AgentState, runtime: Runtime
    ) -> ModelRequest:
        return request

    # Both should create valid middleware instances
    assert isinstance(sync_before, AgentMiddleware)
    assert isinstance(async_before, AgentMiddleware)
    assert isinstance(sync_modify, AgentMiddleware)
    assert isinstance(async_modify, AgentMiddleware)


@pytest.mark.asyncio
async def test_async_decorators_integration() -> None:
    """Test async decorators working together in an agent."""
    call_order = []

    @before_model
    async def track_async_before(state: AgentState, runtime: Runtime) -> None:
        call_order.append("async_before")
        return None

    @modify_model_request
    async def track_async_modify(
        request: ModelRequest, state: AgentState, runtime: Runtime
    ) -> ModelRequest:
        call_order.append("async_modify")
        return request

    @after_model
    async def track_async_after(state: AgentState, runtime: Runtime) -> None:
        call_order.append("async_after")
        return None

    agent = create_agent(
        model=FakeToolCallingModel(),
        middleware=[track_async_before, track_async_modify, track_async_after],
    )
    agent = agent.compile()
    await agent.ainvoke({"messages": [HumanMessage("Hello")]})

    assert call_order == ["async_before", "async_modify", "async_after"]


@pytest.mark.asyncio
async def test_mixed_sync_async_decorators_integration() -> None:
    """Test mixed sync/async decorators working together in an agent."""
    call_order = []

    @before_model
    def track_sync_before(state: AgentState, runtime: Runtime) -> None:
        call_order.append("sync_before")
        return None

    @before_model
    async def track_async_before(state: AgentState, runtime: Runtime) -> None:
        call_order.append("async_before")
        return None

    @modify_model_request
    def track_sync_modify(
        request: ModelRequest, state: AgentState, runtime: Runtime
    ) -> ModelRequest:
        call_order.append("sync_modify")
        return request

    @modify_model_request
    async def track_async_modify(
        request: ModelRequest, state: AgentState, runtime: Runtime
    ) -> ModelRequest:
        call_order.append("async_modify")
        return request

    @after_model
    async def track_async_after(state: AgentState, runtime: Runtime) -> None:
        call_order.append("async_after")
        return None

    @after_model
    def track_sync_after(state: AgentState, runtime: Runtime) -> None:
        call_order.append("sync_after")
        return None

    agent = create_agent(
        model=FakeToolCallingModel(),
        middleware=[
            track_sync_before,
            track_async_before,
            track_sync_modify,
            track_async_modify,
            track_async_after,
            track_sync_after,
        ],
    )
    agent = agent.compile()
    await agent.ainvoke({"messages": [HumanMessage("Hello")]})

    assert call_order == [
        "sync_before",
        "async_before",
        "sync_modify",
        "async_modify",
        "sync_after",
        "async_after",
    ]
