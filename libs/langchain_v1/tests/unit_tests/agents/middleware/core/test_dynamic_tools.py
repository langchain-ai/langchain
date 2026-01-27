"""Tests for dynamic tool registration via middleware.

These tests verify that middleware can dynamically register and handle tools
that are not declared upfront when creating the agent.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def static_tool(value: str) -> str:
    """A static tool that is always available."""
    return f"Static result: {value}"


@tool
def dynamic_tool(value: str) -> str:
    """A dynamically registered tool."""
    return f"Dynamic result: {value}"


@tool
def another_dynamic_tool(x: int, y: int) -> str:
    """Another dynamically registered tool for calculations."""
    return f"Sum: {x + y}"


# -----------------------------------------------------------------------------
# Middleware classes
# -----------------------------------------------------------------------------


class DynamicToolMiddleware(AgentMiddleware):
    """Middleware that dynamically adds and handles a tool (sync and async)."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        updated = request.override(tools=[*request.tools, dynamic_tool])
        return handler(updated)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        updated = request.override(tools=[*request.tools, dynamic_tool])
        return await handler(updated)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] == "dynamic_tool":
            return handler(request.override(tool=dynamic_tool))
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] == "dynamic_tool":
            return await handler(request.override(tool=dynamic_tool))
        return await handler(request)


class MultipleDynamicToolsMiddleware(AgentMiddleware):
    """Middleware that dynamically adds multiple tools (sync and async)."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        updated = request.override(tools=[*request.tools, dynamic_tool, another_dynamic_tool])
        return handler(updated)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        updated = request.override(tools=[*request.tools, dynamic_tool, another_dynamic_tool])
        return await handler(updated)

    def _handle_tool(self, request: ToolCallRequest) -> ToolCallRequest | None:
        """Return updated request if this is a dynamic tool, else None."""
        tool_name = request.tool_call["name"]
        if tool_name == "dynamic_tool":
            return request.override(tool=dynamic_tool)
        if tool_name == "another_dynamic_tool":
            return request.override(tool=another_dynamic_tool)
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        updated = self._handle_tool(request)
        return handler(updated if updated else request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        updated = self._handle_tool(request)
        return await handler(updated if updated else request)


class DynamicToolMiddlewareWithoutHandler(AgentMiddleware):
    """Middleware that adds a dynamic tool but doesn't handle it."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        updated = request.override(tools=[*request.tools, dynamic_tool])
        return handler(updated)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        updated = request.override(tools=[*request.tools, dynamic_tool])
        return await handler(updated)


class ConditionalDynamicToolMiddleware(AgentMiddleware):
    """Middleware that conditionally adds a tool based on state (sync and async)."""

    def _should_add_tool(self, request: ModelRequest) -> bool:
        messages = request.state.get("messages", [])
        return messages and "calculator" in str(messages[-1].content).lower()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        if self._should_add_tool(request):
            request = request.override(tools=[*request.tools, another_dynamic_tool])
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        if self._should_add_tool(request):
            request = request.override(tools=[*request.tools, another_dynamic_tool])
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] == "another_dynamic_tool":
            return handler(request.override(tool=another_dynamic_tool))
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] == "another_dynamic_tool":
            return await handler(request.override(tool=another_dynamic_tool))
        return await handler(request)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def get_tool_messages(result: dict[str, Any]) -> list[ToolMessage]:
    """Extract ToolMessage objects from agent result."""
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


async def invoke_agent(agent: Any, message: str, *, use_async: bool) -> dict[str, Any]:
    """Invoke agent synchronously or asynchronously based on flag."""
    input_data = {"messages": [HumanMessage(message)]}
    config = {"configurable": {"thread_id": "test"}}
    if use_async:
        return await agent.ainvoke(input_data, config)
    # Run sync invoke in thread pool to avoid blocking the event loop
    return await asyncio.to_thread(agent.invoke, input_data, config)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    "tools",
    [
        pytest.param([static_tool], id="with_static_tools"),
        pytest.param([], id="without_static_tools"),
        pytest.param(None, id="with_none_tools"),
    ],
)
async def test_dynamic_tool_basic(*, use_async: bool, tools: list[Any] | None) -> None:
    """Test dynamic tool registration with various static tool configurations."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=tools,  # type: ignore[arg-type]
        middleware=[DynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await invoke_agent(agent, "Use the dynamic tool", use_async=use_async)

    tool_messages = get_tool_messages(result)
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"
    assert "Dynamic result: test" in tool_messages[0].content


@pytest.mark.parametrize("use_async", [False, True])
async def test_multiple_dynamic_tools_with_static(*, use_async: bool) -> None:
    """Test multiple dynamic tools and mixing with static tool calls."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="static_tool", args={"value": "static-call"}, id="1"),
                ToolCall(name="dynamic_tool", args={"value": "first"}, id="2"),
                ToolCall(name="another_dynamic_tool", args={"x": 5, "y": 3}, id="3"),
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[MultipleDynamicToolsMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await invoke_agent(agent, "Use all tools", use_async=use_async)

    tool_messages = get_tool_messages(result)
    assert len(tool_messages) == 3

    tool_results = {m.name: m.content for m in tool_messages}
    assert "Static result: static-call" in tool_results["static_tool"]
    assert "Dynamic result: first" in tool_results["dynamic_tool"]
    assert "Sum: 8" in tool_results["another_dynamic_tool"]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    "tools",
    [
        pytest.param([static_tool], id="with_static_tools"),
        pytest.param([], id="without_static_tools"),
    ],
)
async def test_dynamic_tool_without_handler_raises_error(
    *, use_async: bool, tools: list[Any]
) -> None:
    """Test that a helpful error is raised when dynamic tool is not handled."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[DynamicToolMiddlewareWithoutHandler()],
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(
        ValueError,
        match=r"(?s)Middleware added tools.*Unknown tools:.*dynamic_tool",
    ):
        await invoke_agent(agent, "Use the dynamic tool", use_async=use_async)


@pytest.mark.parametrize("use_async", [False, True])
async def test_conditional_dynamic_tool(*, use_async: bool) -> None:
    """Test that dynamic tools can be conditionally added based on state."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="another_dynamic_tool", args={"x": 10, "y": 20}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[ConditionalDynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await invoke_agent(agent, "I need a calculator to add numbers", use_async=use_async)

    tool_messages = get_tool_messages(result)
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "another_dynamic_tool"
    assert "Sum: 30" in tool_messages[0].content


@pytest.mark.parametrize("use_async", [False, True])
async def test_dynamic_tool_chained_middleware(*, use_async: bool) -> None:
    """Test dynamic tools work with multiple middleware in chain."""
    call_log: list[str] = []

    class LoggingMiddleware(AgentMiddleware):
        def __init__(self, label: str) -> None:
            self._label = label

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            call_log.append(f"{self._label}_model")
            return handler(request)

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            call_log.append(f"{self._label}_model")
            return await handler(request)

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
        ) -> ToolMessage | Command[Any]:
            call_log.append(f"{self._label}_tool")
            return handler(request)

        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        ) -> ToolMessage | Command[Any]:
            call_log.append(f"{self._label}_tool")
            return await handler(request)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "chained"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[LoggingMiddleware("first"), DynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await invoke_agent(agent, "Use the dynamic tool", use_async=use_async)

    tool_messages = get_tool_messages(result)
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"

    # Verify middleware chain was called
    assert "first_model" in call_log
    assert "first_tool" in call_log
