"""Consolidated tests for middleware decorators: before_model, after_model, and wrap_model_call."""

from collections.abc import Awaitable, Callable
from typing import Any, Generic

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from syrupy.assertion import SnapshotAssertion
from typing_extensions import NotRequired

from langchain.agents.factory import _get_can_jump_to, create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    ResponseT,
    after_model,
    before_model,
    dynamic_prompt,
    hook_config,
    wrap_model_call,
    wrap_tool_call,
)
from tests.unit_tests.agents.model import FakeToolCallingModel


class CustomState(AgentState[ResponseT], Generic[ResponseT]):
    """Custom state schema for testing."""

    custom_field: NotRequired[str]


@tool
def test_tool(value: str) -> str:
    """A test tool for middleware testing."""
    return f"Tool result: {value}"


def test_before_model_decorator() -> None:
    """Test before_model decorator with all configuration options."""

    @before_model(
        state_schema=CustomState, tools=[test_tool], can_jump_to=["end"], name="CustomBeforeModel"
    )
    def custom_before_model(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"jump_to": "end"}

    assert isinstance(custom_before_model, AgentMiddleware)
    assert custom_before_model.state_schema == CustomState
    assert custom_before_model.tools == [test_tool]
    assert getattr(custom_before_model.__class__.before_model, "__can_jump_to__", []) == ["end"]
    assert custom_before_model.__class__.__name__ == "CustomBeforeModel"

    result = custom_before_model.before_model({"messages": [HumanMessage("Hello")]}, Runtime())
    assert result == {"jump_to": "end"}


def test_after_model_decorator() -> None:
    """Test after_model decorator with all configuration options."""

    @after_model(
        state_schema=CustomState,
        tools=[test_tool],
        can_jump_to=["model", "end"],
        name="CustomAfterModel",
    )
    def custom_after_model(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
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
        {"messages": [HumanMessage("Hello"), AIMessage("Hi!")]}, Runtime()
    )
    assert result == {"jump_to": "model"}


def test_on_model_call_decorator() -> None:
    """Test wrap_model_call decorator with all configuration options."""

    @wrap_model_call(state_schema=CustomState, tools=[test_tool], name="CustomOnModelCall")
    def custom_on_model_call(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(request.override(system_message=SystemMessage(content="Modified")))

    # Verify all options were applied
    assert isinstance(custom_on_model_call, AgentMiddleware)
    assert custom_on_model_call.state_schema == CustomState
    assert custom_on_model_call.tools == [test_tool]
    assert custom_on_model_call.__class__.__name__ == "CustomOnModelCall"

    # Verify it works
    original_request = ModelRequest(
        model=FakeToolCallingModel(),
        system_prompt="Original",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=None,
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(
            result=[AIMessage(content=f"Handled with prompt: {req.system_prompt}")]
        )

    result = custom_on_model_call.wrap_model_call(original_request, mock_handler)
    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "Handled with prompt: Modified"


def test_all_decorators_integration() -> None:
    """Test all decorators working together in an agent."""
    call_order = []

    @before_model
    def track_before(*_args: Any, **_kwargs: Any) -> None:
        call_order.append("before")

    @wrap_model_call
    def track_on_call(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        call_order.append("on_call")
        return handler(request)

    @after_model
    def track_after(*_args: Any, **_kwargs: Any) -> None:
        call_order.append("after")

    agent = create_agent(
        model=FakeToolCallingModel(), middleware=[track_before, track_on_call, track_after]
    )
    # Agent is already compiled
    agent.invoke({"messages": [HumanMessage("Hello")]})

    assert call_order == ["before", "on_call", "after"]


def test_decorators_use_function_names_as_default() -> None:
    """Test that decorators use function names as default middleware names."""

    @before_model
    def my_before_hook(*_args: Any, **_kwargs: Any) -> None:
        return None

    @wrap_model_call
    def my_on_call_hook(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(request)

    @after_model
    def my_after_hook(*_args: Any, **_kwargs: Any) -> None:
        return None

    # Verify that function names are used as middleware class names
    assert my_before_hook.__class__.__name__ == "my_before_hook"
    assert my_on_call_hook.__class__.__name__ == "my_on_call_hook"
    assert my_after_hook.__class__.__name__ == "my_after_hook"


def test_hook_config_decorator_on_class_method() -> None:
    """Test hook_config decorator on AgentMiddleware class methods."""

    class JumpMiddleware(AgentMiddleware):
        @hook_config(can_jump_to=["end", "model"])
        def before_model(
            self, state: AgentState[Any], runtime: Runtime[None]
        ) -> dict[str, Any] | None:
            if len(state["messages"]) > 5:
                return {"jump_to": "end"}
            return None

        @hook_config(can_jump_to=["tools"])
        def after_model(
            self, state: AgentState[Any], runtime: Runtime[None]
        ) -> dict[str, Any] | None:
            return {"jump_to": "tools"}

    # Verify can_jump_to metadata is preserved
    assert getattr(JumpMiddleware.before_model, "__can_jump_to__", []) == ["end", "model"]
    assert getattr(JumpMiddleware.after_model, "__can_jump_to__", []) == ["tools"]


def test_can_jump_to_with_before_model_decorator() -> None:
    """Test can_jump_to parameter used with before_model decorator."""

    @before_model(can_jump_to=["end"])
    def conditional_before(
        state: AgentState[Any], *_args: Any, **_kwargs: Any
    ) -> dict[str, Any] | None:
        if len(state["messages"]) > 3:
            return {"jump_to": "end"}
        return None

    # Verify middleware was created and has can_jump_to metadata
    assert isinstance(conditional_before, AgentMiddleware)
    assert getattr(conditional_before.__class__.before_model, "__can_jump_to__", []) == ["end"]


def test_can_jump_to_with_after_model_decorator() -> None:
    """Test can_jump_to parameter used with after_model decorator."""

    @after_model(can_jump_to=["model", "end"])
    def conditional_after(
        state: AgentState[Any], *_args: Any, **_kwargs: Any
    ) -> dict[str, Any] | None:
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
    def early_exit(state: AgentState[Any], *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
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
    async def async_before_model(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"custom_field": "async_value"}

    assert isinstance(async_before_model, AgentMiddleware)
    assert async_before_model.state_schema == CustomState
    assert async_before_model.tools == [test_tool]
    assert async_before_model.__class__.__name__ == "AsyncBeforeModel"


def test_async_after_model_decorator() -> None:
    """Test after_model decorator with async function."""

    @after_model(state_schema=CustomState, tools=[test_tool], name="AsyncAfterModel")
    async def async_after_model(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"custom_field": "async_value"}

    assert isinstance(async_after_model, AgentMiddleware)
    assert async_after_model.state_schema == CustomState
    assert async_after_model.tools == [test_tool]
    assert async_after_model.__class__.__name__ == "AsyncAfterModel"


def test_async_on_model_call_decorator() -> None:
    """Test wrap_model_call decorator with async function."""

    @wrap_model_call(state_schema=CustomState, tools=[test_tool], name="AsyncOnModelCall")
    async def async_on_model_call(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(
            request.override(system_message=SystemMessage(content="Modified async"))
        )

    assert isinstance(async_on_model_call, AgentMiddleware)
    assert async_on_model_call.state_schema == CustomState
    assert async_on_model_call.tools == [test_tool]
    assert async_on_model_call.__class__.__name__ == "AsyncOnModelCall"


def test_mixed_sync_async_decorators() -> None:
    """Test decorators with both sync and async functions."""

    @before_model(name="MixedBeforeModel")
    def sync_before(*_args: Any, **_kwargs: Any) -> None:
        return None

    @before_model(name="MixedBeforeModel")
    async def async_before(*_args: Any, **_kwargs: Any) -> None:
        return None

    @wrap_model_call(name="MixedOnModelCall")
    def sync_on_call(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(request)

    @wrap_model_call(name="MixedOnModelCall")
    async def async_on_call(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(request)

    # Both should create valid middleware instances
    assert isinstance(sync_before, AgentMiddleware)
    assert isinstance(async_before, AgentMiddleware)
    assert isinstance(sync_on_call, AgentMiddleware)
    assert isinstance(async_on_call, AgentMiddleware)


async def test_async_decorators_integration() -> None:
    """Test async decorators working together in an agent."""
    call_order = []

    @before_model
    async def track_async_before(*_args: Any, **_kwargs: Any) -> None:
        call_order.append("async_before")

    @wrap_model_call
    async def track_async_on_call(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        call_order.append("async_on_call")
        return await handler(request)

    @after_model
    async def track_async_after(*_args: Any, **_kwargs: Any) -> None:
        call_order.append("async_after")

    agent = create_agent(
        model=FakeToolCallingModel(),
        middleware=[track_async_before, track_async_on_call, track_async_after],
    )
    # Agent is already compiled
    await agent.ainvoke({"messages": [HumanMessage("Hello")]})

    assert call_order == ["async_before", "async_on_call", "async_after"]


async def test_mixed_sync_async_decorators_integration() -> None:
    """Test mixed sync/async decorators working together in an agent."""
    call_order = []

    @before_model
    def track_sync_before(*_args: Any, **_kwargs: Any) -> None:
        call_order.append("sync_before")

    @before_model
    async def track_async_before(*_args: Any, **_kwargs: Any) -> None:
        call_order.append("async_before")

    @wrap_model_call
    async def track_async_on_call(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        call_order.append("async_on_call")
        return await handler(request)

    @wrap_tool_call
    async def track_sync_on_tool_call(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call_order.append("async_on_tool_call")
        return await handler(request)

    @after_model
    async def track_async_after(*_args: Any, **_kwargs: Any) -> None:
        call_order.append("async_after")

    @after_model
    def track_sync_after(*_args: Any, **_kwargs: Any) -> None:
        call_order.append("sync_after")

    agent = create_agent(
        model=FakeToolCallingModel(),
        middleware=[
            track_sync_before,
            track_async_before,
            track_async_on_call,
            track_sync_on_tool_call,
            track_async_after,
            track_sync_after,
        ],
    )
    # Agent is already compiled
    await agent.ainvoke({"messages": [HumanMessage("Hello")]})

    # In async mode, we can automatically delegate to sync middleware for nodes
    # (although we cannot delegate to sync middleware for model call or tool call)

    assert call_order == [
        "sync_before",
        "async_before",
        "async_on_call",
        "sync_after",
        "async_after",
    ]


def test_async_before_model_preserves_can_jump_to() -> None:
    """Test that can_jump_to metadata is preserved for async before_model functions."""

    @before_model(can_jump_to=["end"])
    async def async_conditional_before(
        state: AgentState[Any], *_args: Any, **_kwargs: Any
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
    async def async_conditional_after(
        state: AgentState[Any], *_args: Any, **_kwargs: Any
    ) -> dict[str, Any] | None:
        if state["messages"][-1].content == "retry":
            return {"jump_to": "model"}
        return None

    # Verify middleware was created and has can_jump_to metadata
    assert isinstance(async_conditional_after, AgentMiddleware)
    assert getattr(async_conditional_after.__class__.aafter_model, "__can_jump_to__", []) == [
        "model",
        "end",
    ]


async def test_async_can_jump_to_integration() -> None:
    """Test can_jump_to parameter in a full agent with async middleware."""
    calls = []

    @before_model(can_jump_to=["end"])
    async def async_early_exit(
        state: AgentState[Any], *_args: Any, **_kwargs: Any
    ) -> dict[str, Any] | None:
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
        def before_model(
            self, state: AgentState[Any], runtime: Runtime[None]
        ) -> dict[str, Any] | None:
            return None

    sync_middleware = SyncOnlyMiddleware()
    sync_middleware.tools = []

    # Should return can_jump_to from overridden sync method
    assert _get_can_jump_to(sync_middleware, "before_model") == ["end"]

    # Middleware with only async method overridden
    class AsyncOnlyMiddleware(AgentMiddleware):
        @hook_config(can_jump_to=["model"])
        async def aafter_model(
            self, state: AgentState[Any], runtime: Runtime[None]
        ) -> dict[str, Any] | None:
            return None

    async_middleware = AsyncOnlyMiddleware()
    async_middleware.tools = []

    # Should return can_jump_to from overridden async method
    assert _get_can_jump_to(async_middleware, "after_model") == ["model"]


def test_async_middleware_with_can_jump_to_graph_snapshot(snapshot: SnapshotAssertion) -> None:
    """Test async middleware with can_jump_to graph snapshot.

    Test that async middleware with `can_jump_to` creates correct graph structure with
    conditional edges.
    """

    # Test 1: Async before_model with can_jump_to
    @before_model(can_jump_to=["end"])
    async def async_before_with_jump(
        state: AgentState[Any], *_args: Any, **_kwargs: Any
    ) -> dict[str, Any] | None:
        if len(state["messages"]) > 5:
            return {"jump_to": "end"}
        return None

    agent_async_before = create_agent(
        model=FakeToolCallingModel(), middleware=[async_before_with_jump]
    )

    assert agent_async_before.get_graph().draw_mermaid() == snapshot

    # Test 2: Async after_model with can_jump_to
    @after_model(can_jump_to=["model", "end"])
    async def async_after_with_jump(
        state: AgentState[Any], *_args: Any, **_kwargs: Any
    ) -> dict[str, Any] | None:
        if state["messages"][-1].content == "retry":
            return {"jump_to": "model"}
        return None

    agent_async_after = create_agent(
        model=FakeToolCallingModel(), middleware=[async_after_with_jump]
    )

    assert agent_async_after.get_graph().draw_mermaid() == snapshot

    # Test 3: Multiple async middleware with can_jump_to
    @before_model(can_jump_to=["end"])
    async def async_before_early_exit(*_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        return None

    @after_model(can_jump_to=["model"])
    async def async_after_retry(*_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        return None

    agent_multiple_async = create_agent(
        model=FakeToolCallingModel(),
        middleware=[async_before_early_exit, async_after_retry],
    )

    assert agent_multiple_async.get_graph().draw_mermaid() == snapshot

    # Test 4: Mixed sync and async middleware with can_jump_to
    @before_model(can_jump_to=["end"])
    def sync_before_with_jump(*_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        return None

    @after_model(can_jump_to=["model", "end"])
    async def async_after_with_jumps(*_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
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
        model=FakeToolCallingModel(),
        system_prompt="Original",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=None,
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content=req.system_prompt)])

    result = my_prompt.wrap_model_call(original_request, mock_handler)
    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "Dynamic test prompt"


def test_dynamic_prompt_uses_state() -> None:
    """Test that dynamic_prompt can use state information."""

    @dynamic_prompt
    def custom_prompt(request: ModelRequest) -> str:
        msg_count = len(request.state["messages"])
        return f"Prompt with {msg_count} messages"

    # Verify it uses state correctly
    original_request = ModelRequest(
        model=FakeToolCallingModel(),
        system_prompt="Original",
        messages=[HumanMessage("Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello"), HumanMessage("World")]},
        runtime=None,
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content=req.system_prompt)])

    result = custom_prompt.wrap_model_call(original_request, mock_handler)
    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "Prompt with 2 messages"


def test_dynamic_prompt_integration() -> None:
    """Test dynamic_prompt decorator in a full agent."""
    prompt_calls = 0

    @dynamic_prompt
    def context_aware_prompt(request: ModelRequest) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return "you are a helpful assistant."

    agent = create_agent(model=FakeToolCallingModel(), middleware=[context_aware_prompt])
    # Agent is already compiled

    result = agent.invoke({"messages": [HumanMessage("Hello")]})

    assert prompt_calls == 1
    assert result["messages"][-1].content == "you are a helpful assistant.-Hello"


def test_async_dynamic_prompt_decorator() -> None:
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
        return "Async assistant."

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
    # since they're both wrap_model_call hooks composed in sequence
    agent = create_agent(model=FakeToolCallingModel(), middleware=[first_prompt, second_prompt])
    # Agent is already compiled

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert result["messages"][-1].content == "Second prompt.-Hello"


def test_async_dynamic_prompt_skipped_on_sync_invoke() -> None:
    """Test async dynamic_prompt skipped on sync invoke.

    Test that async `dynamic_prompt` raises `NotImplementedError` when invoked via sync
    path (.invoke).

    When an async-only middleware is defined, it cannot be called from the sync path.
    The framework will raise NotImplementedError when trying to invoke the sync method.
    """
    calls = []

    @dynamic_prompt
    async def async_only_prompt(request: ModelRequest) -> str:
        calls.append("async_prompt")
        return "Async prompt"

    agent = create_agent(model=FakeToolCallingModel(), middleware=[async_only_prompt])

    # Async-only middleware raises NotImplementedError in sync path
    with pytest.raises(NotImplementedError):
        agent.invoke({"messages": [HumanMessage("Hello")]})

    # The async prompt was not called
    assert calls == []


async def test_sync_dynamic_prompt_on_async_invoke() -> None:
    """Test that sync dynamic_prompt works when invoked via async path (.ainvoke).

    When a sync middleware is defined with @dynamic_prompt, it automatically creates
    both sync and async implementations. The async implementation delegates to the
    sync function, allowing the middleware to work in both sync and async contexts.
    """
    calls = []

    @dynamic_prompt
    def sync_prompt(request: ModelRequest) -> str:
        calls.append("sync_prompt")
        return "Sync prompt"

    agent = create_agent(model=FakeToolCallingModel(), middleware=[sync_prompt])

    # Sync dynamic_prompt now works in async path via delegation
    result = await agent.ainvoke({"messages": [HumanMessage("Hello")]})

    # The sync prompt function was called via async delegation
    assert calls == ["sync_prompt"]
    # The model executed with the custom prompt
    assert result["messages"][-1].content == "Sync prompt-Hello"
