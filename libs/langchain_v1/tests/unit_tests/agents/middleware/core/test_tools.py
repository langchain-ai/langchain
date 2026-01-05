"""Test Middleware handling of tools in agents."""

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langgraph.prebuilt.tool_node import ToolNode

from langchain.agents.factory import create_agent
from langchain.agents.middleware import ModelResponse
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest
from tests.unit_tests.agents.model import FakeToolCallingModel


def test_model_request_tools_are_base_tools() -> None:
    """Test that ModelRequest.tools contains BaseTool objects."""
    captured_requests: list[ModelRequest] = []

    @tool
    def search_tool(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    @tool
    def calculator(expression: str) -> str:
        """Calculate a mathematical expression."""
        return f"Result: {expression}"

    class RequestCapturingMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            captured_requests.append(request)
            return handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[search_tool, calculator],
        system_prompt="You are a helpful assistant.",
        middleware=[RequestCapturingMiddleware()],
    )

    agent.invoke({"messages": [HumanMessage("Hello")]})

    # Verify that at least one request was captured
    assert len(captured_requests) > 0

    # Check that tools in the request are BaseTool objects
    request = captured_requests[0]
    assert isinstance(request.tools, list)
    assert len(request.tools) == 2

    tools = []
    for t in request.tools:
        assert isinstance(t, BaseTool)
        tools.append(t.name)
    assert set(tools) == {
        "search_tool",
        "calculator",
    }


def test_middleware_can_modify_tools() -> None:
    """Test that middleware can modify the list of tools in ModelRequest."""

    @tool
    def tool_a(value: str) -> str:
        """Tool A."""
        return "A"

    @tool
    def tool_b(value: str) -> str:
        """Tool B."""
        return "B"

    @tool
    def tool_c(value: str) -> str:
        """Tool C."""
        return "C"

    class ToolFilteringMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            # Only allow tool_a and tool_b
            filtered_tools = [
                t for t in request.tools if hasattr(t, "name") and t.name in {"tool_a", "tool_b"}
            ]
            return handler(request.override(tools=filtered_tools))

    # Model will try to call tool_a
    model = FakeToolCallingModel(
        tool_calls=[[{"args": {"input": "test"}, "id": "1", "name": "tool_a"}], []]
    )

    agent = create_agent(
        model=model,
        tools=[tool_a, tool_b, tool_c],
        system_prompt="You are a helpful assistant.",
        middleware=[ToolFilteringMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("Use tool_a")]})

    # Verify that the tool was executed successfully
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "tool_a"


def test_unknown_tool_raises_error() -> None:
    """Test that using an unknown tool in ModelRequest raises a clear error."""

    @tool
    def known_tool(value: str) -> str:
        """A known tool."""
        return "result"

    @tool
    def unknown_tool(value: str) -> str:
        """An unknown tool not passed to create_agent."""
        return "unknown"

    class BadMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            # Add an unknown tool
            return handler(request.override(tools=[*request.tools, unknown_tool]))

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[known_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[BadMiddleware()],
    )

    with pytest.raises(ValueError, match="Middleware returned unknown tool names"):
        agent.invoke({"messages": [HumanMessage("Hello")]})


def test_middleware_can_add_and_remove_tools() -> None:
    """Test that middleware can dynamically add/remove tools based on state."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Search results for: {query}"

    @tool
    def admin_tool(command: str) -> str:
        """Admin-only tool."""
        return f"Admin: {command}"

    class AdminState(AgentState[Any]):
        is_admin: bool

    class ConditionalToolMiddleware(AgentMiddleware[AdminState]):
        state_schema = AdminState

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            # Remove admin_tool if not admin
            if not request.state.get("is_admin", False):
                filtered_tools = [
                    t for t in request.tools if hasattr(t, "name") and t.name != "admin_tool"
                ]
                request = request.override(tools=filtered_tools)
            return handler(request)

    model = FakeToolCallingModel()

    agent = create_agent(
        model=model,
        tools=[search, admin_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[ConditionalToolMiddleware()],
    )

    # Test non-admin user - should not have access to admin_tool
    # We can't directly inspect the bound model, but we can verify the agent runs
    result = agent.invoke({"messages": [HumanMessage("Hello")], "is_admin": False})
    assert "messages" in result

    # Test admin user - should have access to all tools
    result = agent.invoke({"messages": [HumanMessage("Hello")], "is_admin": True})
    assert "messages" in result


def test_empty_tools_list_is_valid() -> None:
    """Test that middleware can set tools to an empty list."""

    @tool
    def some_tool(value: str) -> str:
        """Some tool."""
        return "result"

    class NoToolsMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            # Remove all tools
            request = request.override(tools=[])
            return handler(request)

    model = FakeToolCallingModel()

    agent = create_agent(
        model=model,
        tools=[some_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[NoToolsMiddleware()],
    )

    # Should run without error even with no tools
    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert "messages" in result


def test_tools_preserved_across_multiple_middleware() -> None:
    """Test that tool modifications by one middleware are visible to the next."""
    modification_order: list[list[str]] = []

    @tool
    def tool_a(value: str) -> str:
        """Tool A."""
        return "A"

    @tool
    def tool_b(value: str) -> str:
        """Tool B."""
        return "B"

    @tool
    def tool_c(value: str) -> str:
        """Tool C."""
        return "C"

    class FirstMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            tools = []
            filtered_tools: list[BaseTool | dict[str, Any]] = []
            for t in request.tools:
                assert isinstance(t, BaseTool)
                tools.append(t.name)
                # Remove tool_c
                if t.name != "tool_c":
                    filtered_tools.append(t)
            modification_order.append(tools)
            # Remove tool_c
            request = request.override(tools=filtered_tools)
            return handler(request)

    class SecondMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            tools = []
            filtered_tools: list[BaseTool | dict[str, Any]] = []
            for t in request.tools:
                assert isinstance(t, BaseTool)
                # Should not see tool_c here
                assert t.name != "tool_c"
                tools.append(t.name)
                # Remove tool_b
                if t.name != "tool_b":
                    filtered_tools.append(t)
            modification_order.append(tools)
            request = request.override(tools=filtered_tools)
            return handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[tool_a, tool_b, tool_c],
        system_prompt="You are a helpful assistant.",
        middleware=[FirstMiddleware(), SecondMiddleware()],
    )

    agent.invoke({"messages": [HumanMessage("Hello")]})

    # Verify the modification sequence
    assert len(modification_order) == 2
    # First middleware sees all three tools
    assert set(modification_order[0]) == {"tool_a", "tool_b", "tool_c"}
    # Second middleware sees tool_c removed
    assert set(modification_order[1]) == {"tool_a", "tool_b"}


def test_middleware_with_additional_tools() -> None:
    """Test middleware that provides additional tools via tools attribute."""

    @tool
    def base_tool(value: str) -> str:
        """Base tool."""
        return "base"

    @tool
    def middleware_tool(value: str) -> str:
        """Tool provided by middleware."""
        return "middleware"

    class ToolProvidingMiddleware(AgentMiddleware):
        tools = (middleware_tool,)

    # Model calls the middleware-provided tool
    model = FakeToolCallingModel(
        tool_calls=[
            [{"args": {"value": "test"}, "id": "1", "name": "middleware_tool"}],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[base_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[ToolProvidingMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("Use middleware tool")]})

    # Verify that the middleware tool was executed
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "middleware_tool"
    assert isinstance(tool_messages[0].content, str)
    assert "middleware" in tool_messages[0].content.lower()


def test_tool_node_not_accepted() -> None:
    """Test that passing a ToolNode instance to create_agent raises an error."""

    @tool
    def some_tool(value: str) -> str:
        """Some tool."""
        return "result"

    tool_node = ToolNode([some_tool])

    with pytest.raises(TypeError, match="'ToolNode' object is not iterable"):
        create_agent(
            model=FakeToolCallingModel(),
            tools=tool_node,  # type: ignore[arg-type]
            system_prompt="You are a helpful assistant.",
        )
