"""Tests for async middleware decorators: abefore_model, aafter_model, and amodify_model_request."""

import asyncio
from typing import Any
from typing_extensions import NotRequired

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.types import Command

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    abefore_model,
    aafter_model,
    amodify_model_request,
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


def test_abefore_model_decorator() -> None:
    """Test abefore_model decorator with all configuration options."""

    @abefore_model(
        state_schema=CustomState, tools=[test_tool], jump_to=["end"], name="CustomAsyncBeforeModel"
    )
    async def custom_async_before_model(state: CustomState) -> dict[str, Any]:
        await asyncio.sleep(0.001)  # Simulate async work
        return {"jump_to": "end"}

    assert isinstance(custom_async_before_model, AgentMiddleware)
    assert custom_async_before_model.state_schema == CustomState
    assert custom_async_before_model.tools == [test_tool]
    assert custom_async_before_model.before_model_jump_to == ["end"]
    assert custom_async_before_model.__class__.__name__ == "CustomAsyncBeforeModel"

    # Test that the async method was set
    assert hasattr(custom_async_before_model, "abefore_model")
    result = asyncio.run(
        custom_async_before_model.abefore_model({"messages": [HumanMessage("Hello")]})
    )
    assert result == {"jump_to": "end"}


def test_aafter_model_decorator() -> None:
    """Test aafter_model decorator with all configuration options."""

    @aafter_model(
        state_schema=CustomState,
        tools=[test_tool],
        jump_to=["model", "end"],
        name="CustomAsyncAfterModel",
    )
    async def custom_async_after_model(state: CustomState) -> dict[str, Any]:
        await asyncio.sleep(0.001)  # Simulate async work
        return {"jump_to": "model"}

    # Verify all options were applied
    assert isinstance(custom_async_after_model, AgentMiddleware)
    assert custom_async_after_model.state_schema == CustomState
    assert custom_async_after_model.tools == [test_tool]
    assert custom_async_after_model.after_model_jump_to == ["model", "end"]
    assert custom_async_after_model.__class__.__name__ == "CustomAsyncAfterModel"

    # Verify it works
    result = asyncio.run(
        custom_async_after_model.aafter_model(
            {"messages": [HumanMessage("Hello"), AIMessage("Hi!")]}
        )
    )
    assert result == {"jump_to": "model"}


def test_amodify_model_request_decorator() -> None:
    """Test amodify_model_request decorator with all configuration options."""

    @amodify_model_request(
        state_schema=CustomState, tools=[test_tool], name="CustomAsyncModifyRequest"
    )
    async def custom_async_modify_request(
        request: ModelRequest, state: CustomState
    ) -> ModelRequest:
        await asyncio.sleep(0.001)  # Simulate async work
        request.system_prompt = "Async Modified"
        return request

    # Verify all options were applied
    assert isinstance(custom_async_modify_request, AgentMiddleware)
    assert custom_async_modify_request.state_schema == CustomState
    assert custom_async_modify_request.tools == [test_tool]
    assert custom_async_modify_request.__class__.__name__ == "CustomAsyncModifyRequest"

    # Verify it works
    original_request = ModelRequest(
        model="test-model",
        system_prompt="Original",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
    )
    result = asyncio.run(
        custom_async_modify_request.amodify_model_request(
            original_request, {"messages": [HumanMessage("Hello")]}
        )
    )
    assert result.system_prompt == "Async Modified"


def test_all_async_decorators_integration() -> None:
    """Test all three async decorators working together in an agent."""
    call_order = []

    @abefore_model
    async def track_async_before(state: AgentState) -> None:
        await asyncio.sleep(0.001)  # Simulate async work
        call_order.append("async_before")
        return None

    @amodify_model_request
    async def track_async_modify(request: ModelRequest, state: AgentState) -> ModelRequest:
        await asyncio.sleep(0.001)  # Simulate async work
        call_order.append("async_modify")
        return request

    @aafter_model
    async def track_async_after(state: AgentState) -> None:
        await asyncio.sleep(0.001)  # Simulate async work
        call_order.append("async_after")
        return None

    agent = create_agent(
        model=FakeToolCallingModel(),
        middleware=[track_async_before, track_async_modify, track_async_after],
    )
    agent = agent.compile()

    async def run_test():
        result = await agent.ainvoke({"messages": [HumanMessage("Hello")]})
        return result

    asyncio.run(run_test())
    assert call_order == ["async_before", "async_modify", "async_after"]


def test_mixed_sync_async_middleware() -> None:
    """Test mixing sync and async middleware in the same agent."""
    call_order = []

    # Import sync decorators
    from langchain.agents.middleware.types import before_model, modify_model_request, after_model

    @before_model
    def sync_before(state: AgentState) -> None:
        call_order.append("sync_before")
        return None

    @abefore_model
    async def async_before(state: AgentState) -> None:
        await asyncio.sleep(0.001)  # Simulate async work
        call_order.append("async_before")
        return None

    @modify_model_request
    def sync_modify(request: ModelRequest, state: AgentState) -> ModelRequest:
        call_order.append("sync_modify")
        return request

    @amodify_model_request
    async def async_modify(request: ModelRequest, state: AgentState) -> ModelRequest:
        await asyncio.sleep(0.001)  # Simulate async work
        call_order.append("async_modify")
        return request

    @after_model
    def sync_after(state: AgentState) -> None:
        call_order.append("sync_after")
        return None

    @aafter_model
    async def async_after(state: AgentState) -> None:
        await asyncio.sleep(0.001)  # Simulate async work
        call_order.append("async_after")
        return None

    agent = create_agent(
        model=FakeToolCallingModel(),
        middleware=[sync_before, async_before, sync_modify, async_modify, sync_after, async_after],
    )
    agent = agent.compile()

    async def run_test():
        result = await agent.ainvoke({"messages": [HumanMessage("Hello")]})
        return result

    asyncio.run(run_test())

    # Both sync and async middleware should have run
    # Order: before middlewares (sync_before, async_before), modify middlewares (sync_modify, async_modify),
    # after middlewares (async_after, sync_after) - note: after hooks run in reverse order
    expected_calls = [
        "sync_before",
        "async_before",
        "sync_modify",
        "async_modify",
        "async_after",
        "sync_after",
    ]
    assert call_order == expected_calls


def test_async_decorators_use_function_names_as_default() -> None:
    """Test that async decorators use function names as default middleware names."""

    @abefore_model
    async def my_async_before_hook(state: AgentState) -> None:
        return None

    @amodify_model_request
    async def my_async_modify_hook(request: ModelRequest, state: AgentState) -> ModelRequest:
        return request

    @aafter_model
    async def my_async_after_hook(state: AgentState) -> None:
        return None

    # Verify that function names are used as middleware class names
    assert my_async_before_hook.__class__.__name__ == "my_async_before_hook"
    assert my_async_modify_hook.__class__.__name__ == "my_async_modify_hook"
    assert my_async_after_hook.__class__.__name__ == "my_async_after_hook"


def test_async_with_runtime_context() -> None:
    """Test async decorators that use runtime context."""

    @abefore_model
    async def async_before_with_runtime(state: AgentState, runtime) -> dict[str, Any]:
        await asyncio.sleep(0.001)  # Simulate async work
        # Use runtime context in some way
        context_info = getattr(runtime, "context", {})
        return {"custom_field": f"processed_with_runtime_{len(context_info)}"}

    @amodify_model_request
    async def async_modify_with_runtime(
        request: ModelRequest, state: AgentState, runtime
    ) -> ModelRequest:
        await asyncio.sleep(0.001)  # Simulate async work
        # Modify request based on runtime context
        request.system_prompt = f"Runtime context available: {runtime is not None}"
        return request

    @aafter_model
    async def async_after_with_runtime(state: AgentState, runtime) -> None:
        await asyncio.sleep(0.001)  # Simulate async work
        # Process state with runtime
        return None

    # Test that these can be instantiated (runtime context validation will happen at execution)
    assert isinstance(async_before_with_runtime, AgentMiddleware)
    assert isinstance(async_modify_with_runtime, AgentMiddleware)
    assert isinstance(async_after_with_runtime, AgentMiddleware)


def test_sync_execution_with_async_only_middleware_error() -> None:
    """Test that sync execution properly errors when encountering async-only middleware."""
    import pytest

    @amodify_model_request
    async def async_only_modify(request: ModelRequest, state: AgentState) -> ModelRequest:
        await asyncio.sleep(0.001)  # Simulate async work
        request.system_prompt = "Modified by async-only middleware"
        return request

    # Create an agent with async-only middleware
    agent = create_agent(model=FakeToolCallingModel(), middleware=[async_only_modify])
    agent = agent.compile()

    # Sync execution should raise an error
    with pytest.raises(ValueError, match="only has async modify_model_request hook"):
        agent.invoke({"messages": [HumanMessage("Hello")]})

    # Async execution should work fine
    async def run_async_test():
        result = await agent.ainvoke({"messages": [HumanMessage("Hello")]})
        return result

    # This should not raise an error
    result = asyncio.run(run_async_test())
    assert result is not None


def test_mixed_middleware_execution_contexts() -> None:
    """Test that mixed sync/async middleware works in both execution contexts."""
    call_order = []

    # Import sync decorators
    from langchain.agents.middleware.types import modify_model_request

    @modify_model_request
    def sync_only_modify(request: ModelRequest, state: AgentState) -> ModelRequest:
        call_order.append("sync_only_modify")
        return request

    @amodify_model_request
    async def async_only_modify(request: ModelRequest, state: AgentState) -> ModelRequest:
        await asyncio.sleep(0.001)
        call_order.append("async_only_modify")
        return request

    # Create agent with both sync-only and async-only middleware
    agent = create_agent(
        model=FakeToolCallingModel(), middleware=[sync_only_modify, async_only_modify]
    )
    agent = agent.compile()

    # Test async execution - should work and call both middleware
    async def run_async_test():
        call_order.clear()
        result = await agent.ainvoke({"messages": [HumanMessage("Hello")]})
        return result

    asyncio.run(run_async_test())
    assert "sync_only_modify" in call_order
    assert "async_only_modify" in call_order

    # Test sync execution - should fail due to async-only middleware
    import pytest

    call_order.clear()
    with pytest.raises(ValueError, match="only has async modify_model_request hook"):
        agent.invoke({"messages": [HumanMessage("Hello")]})


def test_error_handling_for_missing_implementations() -> None:
    """Test error handling when middleware validation fails."""
    import pytest

    # Create a custom middleware with no implementations
    class BrokenMiddleware(AgentMiddleware):
        def __init__(self):
            self.tools = []

    broken_middleware = BrokenMiddleware()

    # This should not cause immediate error during agent creation
    # The validation logic should properly filter out middleware without implementations
    agent = create_agent(model=FakeToolCallingModel(), middleware=[broken_middleware])
    agent = agent.compile()

    # Execution should work normally since the broken middleware has no hooks
    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert result is not None
