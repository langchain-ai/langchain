"""Consolidated tests for middleware decorators: before_model, after_model, and modify_model_request."""

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
    dynamic_prompt,
    modify_model_request,
    hook_config,
)
from langchain.agents.factory import create_agent, _get_can_jump_to
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
        state_schema=CustomState, tools=[test_tool], can_jump_to=["end"], name="CustomBeforeModel"
    )
    def custom_before_model(state: CustomState, runtime: Runtime) -> dict[str, Any]:
        return {"jump_to": "end"}

    assert isinstance(custom_before_model, AgentMiddleware)
    assert custom_before_model.state_schema == CustomState
    assert custom_before_model.tools == [test_tool]
    assert getattr(custom_before_model.__class__.before_model, "__can_jump_to__", []) == ["end"]
    assert custom_before_model.__class__.__name__ == "CustomBeforeModel"

    result = custom_before_model.before_model({"messages": [HumanMessage("Hello")]}, None)
    assert result == {"jump_to": "end"}


def test_after_model_decorator() -> None:
    """Test after_model decorator with all configuration options."""

    @after_model(
        state_schema=CustomState,
        tools=[test_tool],
        can_jump_to=["model", "end"],
        name="CustomAfterModel",
    )
    def custom_after_model(state: CustomState, runtime: Runtime) -> dict[str, Any]:
        return {"jump_to": "model"}

    # Verify all options were applied
    assert isinstance(custom_after_model, AgentMiddleware)
    assert custom_after_model.state_schema == CustomState
    assert custom_after_model.tools == [test_tool]
    assert getattr(custom_after_model.__class__.after_model, "__can_jump_to__", []) == [
        "model",
        "end",
    ]
    assert custom_after_model.__class__.__name__ == "CustomAfterModel"

    # Verify it works
    result = custom_after_model.after_model(
        {"messages": [HumanMessage("Hello"), AIMessage("Hi!")]}, None
    )
    assert result == {"jump_to": "model"}


def test_modify_model_request_decorator() -> None:
    """Test modify_model_request decorator with all configuration options."""

    @modify_model_request(state_schema=CustomState, tools=[test_tool], name="CustomModifyRequest")
    def custom_modify_request(request: ModelRequest) -> ModelRequest:
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
        state={"messages": [HumanMessage("Hello")]},
        runtime=None,
    )
    result = custom_modify_request.modify_model_request(original_request)
    assert result.system_prompt == "Modified"


def test_all_decorators_integration() -> None:
    """Test all three decorators working together in an agent."""
    call_order = []

    @before_model
    def track_before(state: AgentState, runtime: Runtime) -> None:
        call_order.append("before")
        return None

    @modify_model_request
    def track_modify(request: ModelRequest) -> ModelRequest:
        call_order.append("modify")
        return request

    @after_model
    def track_after(state: AgentState, runtime: Runtime) -> None:
        call_order.append("after")
        return None

    agent = create_agent(
        model=FakeToolCallingModel(), middleware=[track_before, track_modify, track_after]
    )
    # Agent is already compiled
    agent.invoke({"messages": [HumanMessage("Hello")]})

    assert call_order == ["before", "modify", "after"]


def test_decorators_use_function_names_as_default() -> None:
    """Test that decorators use function names as default middleware names."""

    @before_model
    def my_before_hook(state: AgentState, runtime: Runtime) -> None:
        return None

    @modify_model_request
    def my_modify_hook(request: ModelRequest) -> ModelRequest:
        return request

    @after_model
    def my_after_hook(state: AgentState, runtime: Runtime) -> None:
        return None

    # Verify that function names are used as middleware class names
    assert my_before_hook.__class__.__name__ == "my_before_hook"
    assert my_modify_hook.__class__.__name__ == "my_modify_hook"
    assert my_after_hook.__class__.__name__ == "my_after_hook"


def test_hook_config_decorator_on_class_method() -> None:
    """Test hook_config decorator on AgentMiddleware class methods."""

    class JumpMiddleware(AgentMiddleware):
        @hook_config(can_jump_to=["end", "model"])
        def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            if len(state["messages"]) > 5:
                return {"jump_to": "end"}
            return None

        @hook_config(can_jump_to=["tools"])
        def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            return {"jump_to": "tools"}

    # Verify can_jump_to metadata is preserved
    assert getattr(JumpMiddleware.before_model, "__can_jump_to__", []) == ["end", "model"]
    assert getattr(JumpMiddleware.after_model, "__can_jump_to__", []) == ["tools"]


def test_can_jump_to_with_before_model_decorator() -> None:
    """Test can_jump_to parameter used with before_model decorator."""

    @before_model(can_jump_to=["end"])
    def conditional_before(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if len(state["messages"]) > 3:
            return {"jump_to": "end"}
        return None

    # Verify middleware was created and has can_jump_to metadata
    assert isinstance(conditional_before, AgentMiddleware)
    assert getattr(conditional_before.__class__.before_model, "__can_jump_to__", []) == ["end"]


def test_can_jump_to_with_after_model_decorator() -> None:
    """Test can_jump_to parameter used with after_model decorator."""

    @after_model(can_jump_to=["model", "end"])
    def conditional_after(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if state["messages"][-1].content == "retry":
            return {"jump_to": "model"}
        return None

    # Verify middleware was created and has can_jump_to metadata
    assert isinstance(conditional_after, AgentMiddleware)
    assert getattr(conditional_after.__class__.after_model, "__can_jump_to__", []) == [
        "model",
        "end",
    ]


def test_can_jump_to_integration() -> None:
    """Test can_jump_to parameter in a full agent."""
    calls = []

    @before_model(can_jump_to=["end"])
    def early_exit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        calls.append("early_exit")
        if state["messages"][0].content == "exit":
            return {"jump_to": "end"}
        return None

    agent = create_agent(model=FakeToolCallingModel(), middleware=[early_exit])
    # Agent is already compiled

    # Test with early exit
    result = agent.invoke({"messages": [HumanMessage("exit")]})
    assert calls == ["early_exit"]
    assert len(result["messages"]) == 1

    # Test without early exit
    calls.clear()
    result = agent.invoke({"messages": [HumanMessage("hello")]})
    assert calls == ["early_exit"]
    assert len(result["messages"]) > 1


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
    async def async_modify_request(request: ModelRequest) -> ModelRequest:
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
    def sync_modify(request: ModelRequest) -> ModelRequest:
        return request

    @modify_model_request(name="MixedModifyRequest")
    async def async_modify(request: ModelRequest) -> ModelRequest:
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
    async def track_async_modify(request: ModelRequest) -> ModelRequest:
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
    # Agent is already compiled
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
    def track_sync_modify(request: ModelRequest) -> ModelRequest:
        call_order.append("sync_modify")
        return request

    @modify_model_request
    async def track_async_modify(request: ModelRequest) -> ModelRequest:
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
    # Agent is already compiled
    await agent.ainvoke({"messages": [HumanMessage("Hello")]})

    assert call_order == [
        "sync_before",
        "async_before",
        "sync_modify",
        "async_modify",
        "sync_after",
        "async_after",
    ]


def test_async_before_model_preserves_can_jump_to() -> None:
    """Test that can_jump_to metadata is preserved for async before_model functions."""

    @before_model(can_jump_to=["end"])
    async def async_conditional_before(
        state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if len(state["messages"]) > 3:
            return {"jump_to": "end"}
        return None

    # Verify middleware was created and has can_jump_to metadata
    assert isinstance(async_conditional_before, AgentMiddleware)
    assert getattr(async_conditional_before.__class__.abefore_model, "__can_jump_to__", []) == [
        "end"
    ]


def test_async_after_model_preserves_can_jump_to() -> None:
    """Test that can_jump_to metadata is preserved for async after_model functions."""

    @after_model(can_jump_to=["model", "end"])
    async def async_conditional_after(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if state["messages"][-1].content == "retry":
            return {"jump_to": "model"}
        return None

    # Verify middleware was created and has can_jump_to metadata
    assert isinstance(async_conditional_after, AgentMiddleware)
    assert getattr(async_conditional_after.__class__.aafter_model, "__can_jump_to__", []) == [
        "model",
        "end",
    ]


@pytest.mark.asyncio
async def test_async_can_jump_to_integration() -> None:
    """Test can_jump_to parameter in a full agent with async middleware."""
    calls = []

    @before_model(can_jump_to=["end"])
    async def async_early_exit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        calls.append("async_early_exit")
        if state["messages"][0].content == "exit":
            return {"jump_to": "end"}
        return None

    agent = create_agent(model=FakeToolCallingModel(), middleware=[async_early_exit])
    # Agent is already compiled

    # Test with early exit
    result = await agent.ainvoke({"messages": [HumanMessage("exit")]})
    assert calls == ["async_early_exit"]
    assert len(result["messages"]) == 1

    # Test without early exit
    calls.clear()
    result = await agent.ainvoke({"messages": [HumanMessage("hello")]})
    assert calls == ["async_early_exit"]
    assert len(result["messages"]) > 1


def test_get_can_jump_to_no_false_positives() -> None:
    """Test that _get_can_jump_to doesn't return false positives for base class methods."""

    # Middleware with no overridden methods should return empty list
    class EmptyMiddleware(AgentMiddleware):
        pass

    empty_middleware = EmptyMiddleware()
    empty_middleware.tools = []

    # Should not return any jump destinations for base class methods
    assert _get_can_jump_to(empty_middleware, "before_model") == []
    assert _get_can_jump_to(empty_middleware, "after_model") == []


def test_get_can_jump_to_only_overridden_methods() -> None:
    """Test that _get_can_jump_to only checks overridden methods."""

    # Middleware with only sync method overridden
    class SyncOnlyMiddleware(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            return None

    sync_middleware = SyncOnlyMiddleware()
    sync_middleware.tools = []

    # Should return can_jump_to from overridden sync method
    assert _get_can_jump_to(sync_middleware, "before_model") == ["end"]

    # Middleware with only async method overridden
    class AsyncOnlyMiddleware(AgentMiddleware):
        @hook_config(can_jump_to=["model"])
        async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            return None

    async_middleware = AsyncOnlyMiddleware()
    async_middleware.tools = []

    # Should return can_jump_to from overridden async method
    assert _get_can_jump_to(async_middleware, "after_model") == ["model"]


def test_async_middleware_with_can_jump_to_graph_snapshot(snapshot: SnapshotAssertion) -> None:
    """Test that async middleware with can_jump_to creates correct graph structure with conditional edges."""

    # Test 1: Async before_model with can_jump_to
    @before_model(can_jump_to=["end"])
    async def async_before_with_jump(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if len(state["messages"]) > 5:
            return {"jump_to": "end"}
        return None

    agent_async_before = create_agent(
        model=FakeToolCallingModel(), middleware=[async_before_with_jump]
    )

    assert agent_async_before.get_graph().draw_mermaid() == snapshot

    # Test 2: Async after_model with can_jump_to
    @after_model(can_jump_to=["model", "end"])
    async def async_after_with_jump(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if state["messages"][-1].content == "retry":
            return {"jump_to": "model"}
        return None

    agent_async_after = create_agent(
        model=FakeToolCallingModel(), middleware=[async_after_with_jump]
    )

    assert agent_async_after.get_graph().draw_mermaid() == snapshot

    # Test 3: Multiple async middleware with can_jump_to
    @before_model(can_jump_to=["end"])
    async def async_before_early_exit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return None

    @after_model(can_jump_to=["model"])
    async def async_after_retry(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return None

    agent_multiple_async = create_agent(
        model=FakeToolCallingModel(),
        middleware=[async_before_early_exit, async_after_retry],
    )

    assert agent_multiple_async.get_graph().draw_mermaid() == snapshot

    # Test 4: Mixed sync and async middleware with can_jump_to
    @before_model(can_jump_to=["end"])
    def sync_before_with_jump(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return None

    @after_model(can_jump_to=["model", "end"])
    async def async_after_with_jumps(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return None

    agent_mixed = create_agent(
        model=FakeToolCallingModel(),
        middleware=[sync_before_with_jump, async_after_with_jumps],
    )

    assert agent_mixed.get_graph().draw_mermaid() == snapshot


def test_dynamic_prompt_decorator() -> None:
    """Test dynamic_prompt decorator with basic usage."""

    @dynamic_prompt
    def my_prompt(request: ModelRequest) -> str:
        return "Dynamic test prompt"

    assert isinstance(my_prompt, AgentMiddleware)
    assert my_prompt.state_schema == AgentState
    assert my_prompt.tools == []
    assert my_prompt.__class__.__name__ == "my_prompt"

    # Verify it modifies the request correctly
    original_request = ModelRequest(
        model="test-model",
        system_prompt="Original",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=None,
    )
    result = my_prompt.modify_model_request(original_request)
    assert result.system_prompt == "Dynamic test prompt"


def test_dynamic_prompt_uses_state() -> None:
    """Test that dynamic_prompt can use state information."""

    @dynamic_prompt
    def custom_prompt(request: ModelRequest) -> str:
        msg_count = len(request.state["messages"])
        return f"Prompt with {msg_count} messages"

    # Verify it uses state correctly
    original_request = ModelRequest(
        model="test-model",
        system_prompt="Original",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello"), HumanMessage("World")]},
        runtime=None,
    )
    result = custom_prompt.modify_model_request(original_request)
    assert result.system_prompt == "Prompt with 2 messages"


def test_dynamic_prompt_integration() -> None:
    """Test dynamic_prompt decorator in a full agent."""

    prompt_calls = 0

    @dynamic_prompt
    def context_aware_prompt(request: ModelRequest) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return f"you are a helpful assistant."

    agent = create_agent(model=FakeToolCallingModel(), middleware=[context_aware_prompt])
    # Agent is already compiled

    result = agent.invoke({"messages": [HumanMessage("Hello")]})

    assert prompt_calls == 1
    assert result["messages"][-1].content == "you are a helpful assistant.-Hello"


async def test_async_dynamic_prompt_decorator() -> None:
    """Test dynamic_prompt decorator with async function."""

    @dynamic_prompt
    async def async_prompt(request: ModelRequest) -> str:
        return "Async dynamic prompt"

    assert isinstance(async_prompt, AgentMiddleware)
    assert async_prompt.state_schema == AgentState
    assert async_prompt.tools == []
    assert async_prompt.__class__.__name__ == "async_prompt"


async def test_async_dynamic_prompt_integration() -> None:
    """Test async dynamic_prompt decorator in a full agent."""

    prompt_calls = 0

    @dynamic_prompt
    async def async_context_prompt(request: ModelRequest) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return f"Async assistant."

    agent = create_agent(model=FakeToolCallingModel(), middleware=[async_context_prompt])
    # Agent is already compiled

    result = await agent.ainvoke({"messages": [HumanMessage("Hello")]})
    assert prompt_calls == 1
    assert result["messages"][-1].content == "Async assistant.-Hello"


def test_dynamic_prompt_overwrites_system_prompt() -> None:
    """Test that dynamic_prompt overwrites the original system_prompt."""

    @dynamic_prompt
    def override_prompt(request: ModelRequest) -> str:
        return "Overridden prompt."

    agent = create_agent(
        model=FakeToolCallingModel(),
        system_prompt="Original static prompt",
        middleware=[override_prompt],
    )
    # Agent is already compiled

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert result["messages"][-1].content == "Overridden prompt.-Hello"


def test_dynamic_prompt_multiple_in_sequence() -> None:
    """Test multiple dynamic_prompt decorators in sequence (last wins)."""

    @dynamic_prompt
    def first_prompt(request: ModelRequest) -> str:
        return "First prompt."

    @dynamic_prompt
    def second_prompt(request: ModelRequest) -> str:
        return "Second prompt."

    # When used together, the last middleware in the list should win
    # since they're both modify_model_request hooks executed in sequence
    agent = create_agent(model=FakeToolCallingModel(), middleware=[first_prompt, second_prompt])
    # Agent is already compiled

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert result["messages"][-1].content == "Second prompt.-Hello"
