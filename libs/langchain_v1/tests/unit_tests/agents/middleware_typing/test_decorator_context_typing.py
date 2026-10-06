"""Verify context inference across middleware decorators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from typing_extensions import assert_type

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from tests.unit_tests.agents.middleware_typing.test_middleware_typing import (
    CustomAgentState,
    SessionContext,
    UserContext,
)
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.runtime import Runtime
    from langgraph.types import Command


@tool
def echo(value: str) -> str:
    """Return the supplied value."""
    return value


@pytest.mark.parametrize("configured", [False, True])
def test_sync_tool_wrapper_with_context(*, configured: bool) -> None:
    context: UserContext = {"user_id": "123", "user_name": "Alice"}
    seen_contexts: list[Any] = []

    def wrapper(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        seen_contexts.append(request.runtime.context)
        return handler(request)

    bare = wrap_tool_call(wrapper)
    with_state = wrap_tool_call(state_schema=CustomAgentState)(wrapper)
    assert_type(bare, AgentMiddleware[AgentState[Any], Any, Any])
    assert_type(with_state, AgentMiddleware[CustomAgentState, Any, Any])
    create_agent(FakeToolCallingModel(), context_schema=UserContext, middleware=[bare])
    create_agent(FakeToolCallingModel(), context_schema=SessionContext, middleware=[with_state])
    middleware: AgentMiddleware[Any, Any, Any] = with_state if configured else bare
    model = FakeToolCallingModel(
        tool_calls=[[ToolCall(name="echo", args={"value": "ok"}, id="1")], []]
    )
    agent = create_agent(
        model,
        tools=[echo],
        state_schema=CustomAgentState,
        context_schema=UserContext,
        middleware=[middleware],
    )
    result = agent.invoke({"messages": [HumanMessage("Hello")]}, context=context)
    assert seen_contexts == [context]
    assert any(
        isinstance(message, ToolMessage) and message.content == "ok"
        for message in result["messages"]
    )
    assert middleware.state_schema is (CustomAgentState if configured else AgentState)


@pytest.mark.parametrize("configured", [False, True])
async def test_async_tool_wrapper_with_context(*, configured: bool) -> None:
    context: UserContext = {"user_id": "123", "user_name": "Alice"}
    seen_contexts: list[Any] = []

    async def wrapper(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        seen_contexts.append(request.runtime.context)
        return await handler(request)

    bare = wrap_tool_call(wrapper)
    with_state = wrap_tool_call(state_schema=CustomAgentState)(wrapper)
    assert_type(bare, AgentMiddleware[AgentState[Any], Any, Any])
    assert_type(with_state, AgentMiddleware[CustomAgentState, Any, Any])
    create_agent(FakeToolCallingModel(), context_schema=UserContext, middleware=[bare])
    create_agent(FakeToolCallingModel(), context_schema=SessionContext, middleware=[with_state])
    middleware: AgentMiddleware[Any, Any, Any] = with_state if configured else bare
    model = FakeToolCallingModel(
        tool_calls=[[ToolCall(name="echo", args={"value": "ok"}, id="1")], []]
    )
    agent = create_agent(
        model,
        tools=[echo],
        state_schema=CustomAgentState,
        context_schema=UserContext,
        middleware=[middleware],
    )
    result = await agent.ainvoke({"messages": [HumanMessage("Hello")]}, context=context)
    assert seen_contexts == [context]
    assert any(
        isinstance(message, ToolMessage) and message.content == "ok"
        for message in result["messages"]
    )
    assert middleware.state_schema is (CustomAgentState if configured else AgentState)


@pytest.mark.parametrize("configured", [False, True])
def test_other_decorators_preserve_context(*, configured: bool) -> None:
    def node(_state: CustomAgentState, runtime: Runtime[UserContext]) -> None:
        assert_type(runtime.context["user_id"], str)

    async def async_node(_state: CustomAgentState, runtime: Runtime[UserContext]) -> None:
        assert_type(runtime.context["user_id"], str)

    def model_wrapper(
        request: ModelRequest[UserContext],
        handler: Callable[[ModelRequest[UserContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        assert_type(request.runtime.context["user_id"], str)
        return handler(request)

    async def async_model_wrapper(
        request: ModelRequest[UserContext],
        handler: Callable[[ModelRequest[UserContext]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        assert_type(request.runtime.context["user_id"], str)
        return await handler(request)

    def prompt(request: ModelRequest[UserContext]) -> str:
        return request.runtime.context["user_name"]

    async def async_prompt(request: ModelRequest[UserContext]) -> str:
        return request.runtime.context["user_name"]

    for decorator in (before_agent, after_agent, before_model, after_model):
        for callback in (node, async_node):
            middleware = (
                decorator(state_schema=CustomAgentState)(callback)
                if configured
                else decorator(callback)
            )
            assert_type(middleware, AgentMiddleware[CustomAgentState, UserContext, Any])
            create_agent(
                FakeToolCallingModel(), context_schema=UserContext, middleware=[middleware]
            )
            _session_middleware: AgentMiddleware[CustomAgentState, SessionContext, Any]
            _session_middleware = middleware  # type: ignore[assignment]

    for model_callback in (model_wrapper, async_model_wrapper):
        if configured:
            model_middleware = wrap_model_call(state_schema=CustomAgentState)(model_callback)
            assert_type(model_middleware, AgentMiddleware[CustomAgentState, UserContext, Any])
        else:
            bare_model_middleware = wrap_model_call(model_callback)
            assert_type(bare_model_middleware, AgentMiddleware[AgentState[Any], UserContext, Any])
        selected_model: AgentMiddleware[Any, UserContext, Any] = (
            model_middleware if configured else bare_model_middleware
        )
        create_agent(
            FakeToolCallingModel(), context_schema=UserContext, middleware=[selected_model]
        )
        _session_model: AgentMiddleware[Any, SessionContext, Any]
        _session_model = selected_model  # type: ignore[assignment]

    for prompt_callback in (prompt, async_prompt):
        prompt_middleware = (
            dynamic_prompt()(prompt_callback) if configured else dynamic_prompt(prompt_callback)
        )
        assert_type(prompt_middleware, AgentMiddleware[AgentState[Any], UserContext, Any])
        create_agent(
            FakeToolCallingModel(), context_schema=UserContext, middleware=[prompt_middleware]
        )
        _session_prompt: AgentMiddleware[AgentState[Any], SessionContext, Any]
        _session_prompt = prompt_middleware  # type: ignore[assignment]
