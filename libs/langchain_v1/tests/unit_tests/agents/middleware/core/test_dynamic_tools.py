"""Tests for dynamic tool registration via middleware.

These tests verify that middleware can dynamically register and handle tools
that are not declared upfront when creating the agent.
"""

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


class DynamicToolMiddleware(AgentMiddleware):
    """Middleware that dynamically adds and handles a tool."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        # Add the dynamic tool to the model request
        updated = request.override(tools=[*request.tools, dynamic_tool])
        return handler(updated)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        # Handle the dynamic tool by overriding the tool in the request
        if request.tool_call["name"] == "dynamic_tool":
            return handler(request.override(tool=dynamic_tool))
        return handler(request)


class MultipleDynamicToolsMiddleware(AgentMiddleware):
    """Middleware that dynamically adds multiple tools."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        # Add multiple dynamic tools
        updated = request.override(tools=[*request.tools, dynamic_tool, another_dynamic_tool])
        return handler(updated)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_name = request.tool_call["name"]
        # Handle the dynamic tools by overriding the tool in the request
        if tool_name == "dynamic_tool":
            return handler(request.override(tool=dynamic_tool))
        if tool_name == "another_dynamic_tool":
            return handler(request.override(tool=another_dynamic_tool))
        return handler(request)


def test_dynamic_tool_with_static_tools() -> None:
    """Test dynamic tool registration when agent has static tools."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[DynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use the dynamic tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"
    assert "Dynamic result: test" in tool_messages[0].content


def test_dynamic_tool_without_static_tools() -> None:
    """Test dynamic tool registration when agent has no static tools."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "no-static"}, id="1")],
            [],
        ]
    )

    # Create agent with NO static tools - only middleware provides tools
    agent = create_agent(
        model=model,
        tools=[],  # No static tools
        middleware=[DynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use the dynamic tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"
    assert "Dynamic result: no-static" in tool_messages[0].content


def test_dynamic_tool_with_none_tools() -> None:
    """Test dynamic tool registration when tools parameter is None."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "none-tools"}, id="1")],
            [],
        ]
    )

    # Create agent with tools=None
    agent = create_agent(
        model=model,
        tools=None,  # type: ignore[arg-type]
        middleware=[DynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use the dynamic tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"
    assert "Dynamic result: none-tools" in tool_messages[0].content


def test_multiple_dynamic_tools() -> None:
    """Test registering and using multiple dynamic tools."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="dynamic_tool", args={"value": "first"}, id="1"),
                ToolCall(name="another_dynamic_tool", args={"x": 5, "y": 3}, id="2"),
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

    result = agent.invoke(
        {"messages": [HumanMessage("Use both dynamic tools")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2

    # Check both tools were called
    tool_names = {m.name for m in tool_messages}
    assert tool_names == {"dynamic_tool", "another_dynamic_tool"}

    # Check results
    for msg in tool_messages:
        if msg.name == "dynamic_tool":
            assert "Dynamic result: first" in msg.content
        elif msg.name == "another_dynamic_tool":
            assert "Sum: 8" in msg.content


def test_dynamic_tool_mixed_with_static_call() -> None:
    """Test using both static and dynamic tools in the same conversation."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="static_tool", args={"value": "static-call"}, id="1"),
                ToolCall(name="dynamic_tool", args={"value": "dynamic-call"}, id="2"),
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[DynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use both tools")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2

    tool_names = {m.name for m in tool_messages}
    assert tool_names == {"static_tool", "dynamic_tool"}


class DynamicToolMiddlewareWithoutHandler(AgentMiddleware):
    """Middleware that adds a dynamic tool but doesn't handle it in wrap_tool_call."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        # Add the dynamic tool to the model request
        updated = request.override(tools=[*request.tools, dynamic_tool])
        return handler(updated)

    # Note: No wrap_tool_call defined - dynamic tool won't be handled


def test_dynamic_tool_without_handler_raises_error() -> None:
    """Test that a helpful error is raised when dynamic tool is not handled."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[DynamicToolMiddlewareWithoutHandler()],
        checkpointer=InMemorySaver(),
    )

    # Should raise an error because dynamic_tool is not handled
    # The error message should guide the user to define wrap_tool_call
    with pytest.raises(
        ValueError,
        match=r"(?s)Middleware added tools.*Unknown tools:.*dynamic_tool",
    ):
        agent.invoke(
            {"messages": [HumanMessage("Use the dynamic tool")]},
            {"configurable": {"thread_id": "test"}},
        )


class DynamicToolMiddlewareNoStaticWithoutHandler(AgentMiddleware):
    """Middleware that adds a dynamic tool but doesn't handle it, with no static tools."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        updated = request.override(tools=[*request.tools, dynamic_tool])
        return handler(updated)


def test_dynamic_tool_no_static_without_handler_raises_error() -> None:
    """Test error when dynamic tool is not handled and there are no static tools."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[DynamicToolMiddlewareNoStaticWithoutHandler()],
        checkpointer=InMemorySaver(),
    )

    # Should raise an error because there's no wrap_tool_call to handle the dynamic tool
    # The error should guide the user to define wrap_tool_call
    with pytest.raises(
        ValueError,
        match=r"(?s)Middleware added tools.*Unknown tools:.*dynamic_tool",
    ):
        agent.invoke(
            {"messages": [HumanMessage("Use the dynamic tool")]},
            {"configurable": {"thread_id": "test"}},
        )


class ConditionalDynamicToolMiddleware(AgentMiddleware):
    """Middleware that conditionally adds a dynamic tool based on state."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        # Only add the tool if certain conditions are met
        messages = request.state.get("messages", [])
        if messages and "calculator" in str(messages[-1].content).lower():
            updated = request.override(tools=[*request.tools, another_dynamic_tool])
            return handler(updated)
        return handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] == "another_dynamic_tool":
            return handler(request.override(tool=another_dynamic_tool))
        return handler(request)


def test_conditional_dynamic_tool() -> None:
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

    # Request that triggers the dynamic tool
    result = agent.invoke(
        {"messages": [HumanMessage("I need a calculator to add numbers")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "another_dynamic_tool"
    assert "Sum: 30" in tool_messages[0].content


def test_dynamic_tool_with_override_tool() -> None:
    """Test using request.override(tool=...) to provide the tool implementation."""

    class OverrideToolMiddleware(AgentMiddleware):
        """Middleware that uses override to provide tool implementation."""

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            updated = request.override(tools=[*request.tools, dynamic_tool])
            return handler(updated)

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
        ) -> ToolMessage | Command[Any]:
            if request.tool_call["name"] == "dynamic_tool":
                # Use override to provide the tool
                return handler(request.override(tool=dynamic_tool))
            return handler(request)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "override-test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[OverrideToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use the dynamic tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"
    assert "Dynamic result: override-test" in tool_messages[0].content


def test_dynamic_tool_chained_middleware() -> None:
    """Test dynamic tools work with multiple middleware in chain."""
    call_log: list[str] = []

    class FirstMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            call_log.append("first_model")
            return handler(request)

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
        ) -> ToolMessage | Command[Any]:
            call_log.append("first_tool")
            return handler(request)

    class SecondMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            call_log.append("second_model")
            # Add dynamic tool in second middleware
            updated = request.override(tools=[*request.tools, dynamic_tool])
            return handler(updated)

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
        ) -> ToolMessage | Command[Any]:
            call_log.append("second_tool")
            if request.tool_call["name"] == "dynamic_tool":
                return handler(request.override(tool=dynamic_tool))
            return handler(request)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "chained"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[FirstMiddleware(), SecondMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use the dynamic tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"

    # Verify middleware chain was called in correct order
    assert "first_model" in call_log
    assert "second_model" in call_log
    assert "first_tool" in call_log
    assert "second_tool" in call_log


# =============================================================================
# Async Dynamic Tool Tests
# =============================================================================


class AsyncDynamicToolMiddleware(AgentMiddleware):
    """Middleware that dynamically adds and handles a tool using async methods."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # Add the dynamic tool to the model request
        updated = request.override(tools=[*request.tools, dynamic_tool])
        return await handler(updated)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        # Handle the dynamic tool by overriding the tool in the request
        if request.tool_call["name"] == "dynamic_tool":
            return await handler(request.override(tool=dynamic_tool))
        return await handler(request)


class AsyncMultipleDynamicToolsMiddleware(AgentMiddleware):
    """Middleware that dynamically adds multiple tools using async methods."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # Add multiple dynamic tools
        updated = request.override(tools=[*request.tools, dynamic_tool, another_dynamic_tool])
        return await handler(updated)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        tool_name = request.tool_call["name"]
        # Handle the dynamic tools by overriding the tool in the request
        if tool_name == "dynamic_tool":
            return await handler(request.override(tool=dynamic_tool))
        if tool_name == "another_dynamic_tool":
            return await handler(request.override(tool=another_dynamic_tool))
        return await handler(request)


@pytest.mark.asyncio
async def test_async_dynamic_tool_with_static_tools() -> None:
    """Test async dynamic tool registration when agent has static tools."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[AsyncDynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use the dynamic tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"
    assert "Dynamic result: test" in tool_messages[0].content


@pytest.mark.asyncio
async def test_async_dynamic_tool_without_static_tools() -> None:
    """Test async dynamic tool registration when agent has no static tools."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "no-static"}, id="1")],
            [],
        ]
    )

    # Create agent with NO static tools - only middleware provides tools
    agent = create_agent(
        model=model,
        tools=[],  # No static tools
        middleware=[AsyncDynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use the dynamic tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"
    assert "Dynamic result: no-static" in tool_messages[0].content


@pytest.mark.asyncio
async def test_async_dynamic_tool_with_none_tools() -> None:
    """Test async dynamic tool registration when tools parameter is None."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "none-tools"}, id="1")],
            [],
        ]
    )

    # Create agent with tools=None
    agent = create_agent(
        model=model,
        tools=None,  # type: ignore[arg-type]
        middleware=[AsyncDynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use the dynamic tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"
    assert "Dynamic result: none-tools" in tool_messages[0].content


@pytest.mark.asyncio
async def test_async_multiple_dynamic_tools() -> None:
    """Test async registering and using multiple dynamic tools."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="dynamic_tool", args={"value": "first"}, id="1"),
                ToolCall(name="another_dynamic_tool", args={"x": 5, "y": 3}, id="2"),
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[AsyncMultipleDynamicToolsMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use both dynamic tools")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2

    # Check both tools were called
    tool_names = {m.name for m in tool_messages}
    assert tool_names == {"dynamic_tool", "another_dynamic_tool"}

    # Check results
    for msg in tool_messages:
        if msg.name == "dynamic_tool":
            assert "Dynamic result: first" in msg.content
        elif msg.name == "another_dynamic_tool":
            assert "Sum: 8" in msg.content


@pytest.mark.asyncio
async def test_async_dynamic_tool_mixed_with_static_call() -> None:
    """Test async using both static and dynamic tools in the same conversation."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="static_tool", args={"value": "static-call"}, id="1"),
                ToolCall(name="dynamic_tool", args={"value": "dynamic-call"}, id="2"),
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[AsyncDynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use both tools")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2

    tool_names = {m.name for m in tool_messages}
    assert tool_names == {"static_tool", "dynamic_tool"}


class AsyncDynamicToolMiddlewareWithoutHandler(AgentMiddleware):
    """Async middleware that adds a dynamic tool but doesn't handle it."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # Add the dynamic tool to the model request
        updated = request.override(tools=[*request.tools, dynamic_tool])
        return await handler(updated)

    # Note: No awrap_tool_call defined - dynamic tool won't be handled


@pytest.mark.asyncio
async def test_async_dynamic_tool_without_handler_raises_error() -> None:
    """Test that a helpful error is raised when async dynamic tool is not handled."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[AsyncDynamicToolMiddlewareWithoutHandler()],
        checkpointer=InMemorySaver(),
    )

    # Should raise an error because dynamic_tool is not handled
    with pytest.raises(
        ValueError,
        match=r"(?s)Middleware added tools.*Unknown tools:.*dynamic_tool",
    ):
        await agent.ainvoke(
            {"messages": [HumanMessage("Use the dynamic tool")]},
            {"configurable": {"thread_id": "test"}},
        )


class AsyncConditionalDynamicToolMiddleware(AgentMiddleware):
    """Async middleware that conditionally adds a dynamic tool based on state."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # Only add the tool if certain conditions are met
        messages = request.state.get("messages", [])
        if messages and "calculator" in str(messages[-1].content).lower():
            updated = request.override(tools=[*request.tools, another_dynamic_tool])
            return await handler(updated)
        return await handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] == "another_dynamic_tool":
            return await handler(request.override(tool=another_dynamic_tool))
        return await handler(request)


@pytest.mark.asyncio
async def test_async_conditional_dynamic_tool() -> None:
    """Test that async dynamic tools can be conditionally added based on state."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="another_dynamic_tool", args={"x": 10, "y": 20}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[AsyncConditionalDynamicToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    # Request that triggers the dynamic tool
    result = await agent.ainvoke(
        {"messages": [HumanMessage("I need a calculator to add numbers")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "another_dynamic_tool"
    assert "Sum: 30" in tool_messages[0].content


@pytest.mark.asyncio
async def test_async_dynamic_tool_with_override_tool() -> None:
    """Test using request.override(tool=...) to provide async tool implementation."""

    class AsyncOverrideToolMiddleware(AgentMiddleware):
        """Async middleware that uses override to provide tool implementation."""

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            updated = request.override(tools=[*request.tools, dynamic_tool])
            return await handler(updated)

        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        ) -> ToolMessage | Command[Any]:
            if request.tool_call["name"] == "dynamic_tool":
                # Use override to provide the tool
                return await handler(request.override(tool=dynamic_tool))
            return await handler(request)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool", args={"value": "override-test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        middleware=[AsyncOverrideToolMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use the dynamic tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"
    assert "Dynamic result: override-test" in tool_messages[0].content


@pytest.mark.asyncio
async def test_async_dynamic_tool_chained_middleware() -> None:
    """Test async dynamic tools work with multiple middleware in chain."""
    call_log: list[str] = []

    class AsyncFirstMiddleware(AgentMiddleware):
        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            call_log.append("first_model")
            return await handler(request)

        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        ) -> ToolMessage | Command[Any]:
            call_log.append("first_tool")
            return await handler(request)

    class AsyncSecondMiddleware(AgentMiddleware):
        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            call_log.append("second_model")
            # Add dynamic tool in second middleware
            updated = request.override(tools=[*request.tools, dynamic_tool])
            return await handler(updated)

        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        ) -> ToolMessage | Command[Any]:
            call_log.append("second_tool")
            if request.tool_call["name"] == "dynamic_tool":
                return await handler(request.override(tool=dynamic_tool))
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
        middleware=[AsyncFirstMiddleware(), AsyncSecondMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use the dynamic tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool"

    # Verify middleware chain was called in correct order
    assert "first_model" in call_log
    assert "second_model" in call_log
    assert "first_tool" in call_log
    assert "second_tool" in call_log
