"""Tests for middleware handling of tools in agents."""

import pytest

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest
from langchain.agents.factory import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from ..model import FakeToolCallingModel


def test_model_request_tools_are_objects() -> None:
    """Test that ModelRequest.tools contains tool objects (BaseTool | dict)."""
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
        def modify_model_request(
            self, request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> ModelRequest:
            captured_requests.append(request)
            return request

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[search_tool, calculator],
        system_prompt="You are a helpful assistant.",
        middleware=[RequestCapturingMiddleware()],
    )

    agent.invoke({"messages": [HumanMessage("Hello")]})

    # Verify that at least one request was captured
    assert len(captured_requests) > 0

    # Check that tools in the request are tool objects
    request = captured_requests[0]
    assert isinstance(request.tools, list)
    assert len(request.tools) == 2
    tool_names = {t.name for t in request.tools}
    assert tool_names == {"search_tool", "calculator"}


def test_middleware_can_modify_tools() -> None:
    """Test that middleware can modify the list of tools in ModelRequest."""

    @tool
    def tool_a(input: str) -> str:
        """Tool A."""
        return "A"

    @tool
    def tool_b(input: str) -> str:
        """Tool B."""
        return "B"

    @tool
    def tool_c(input: str) -> str:
        """Tool C."""
        return "C"

    class ToolFilteringMiddleware(AgentMiddleware):
        def modify_model_request(
            self, request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> ModelRequest:
            # Only allow tool_a and tool_b
            request.tools = [t for t in request.tools if t.name in ["tool_a", "tool_b"]]
            return request

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


def test_middleware_can_add_custom_tools() -> None:
    """Test that middleware can add custom tool objects to ModelRequest."""

    @tool
    def original_tool(input: str) -> str:
        """Original tool."""
        return "original"

    @tool
    def middleware_tool(input: str) -> str:
        """Middleware-added tool."""
        return "middleware"

    class ToolAddingMiddleware(AgentMiddleware):
        def modify_model_request(
            self, request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> ModelRequest:
            # Add middleware tool to the request
            request.tools = request.tools + [middleware_tool]
            return request

    # Model will try to call middleware_tool
    model = FakeToolCallingModel(
        tool_calls=[[{"args": {"input": "test"}, "id": "1", "name": "middleware_tool"}], []]
    )

    agent = create_agent(
        model=model,
        tools=[original_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[ToolAddingMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("Use middleware_tool")]})

    # Verify that the middleware tool was executed successfully
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "middleware_tool"
    assert tool_messages[0].content == "middleware"


def test_middleware_can_conditionally_add_tools() -> None:
    """Test that middleware can conditionally add tools based on state."""

    @tool
    def admin_tool(input: str) -> str:
        """Admin-only tool."""
        return "admin_result"

    class AdminState(AgentState):
        is_admin: bool = False

    class ConditionalToolMiddleware(AgentMiddleware[AdminState]):
        def modify_model_request(
            self, request: ModelRequest, state: AdminState, runtime: Runtime
        ) -> ModelRequest:
            # Only add admin tool if user is admin
            if state.get("is_admin", False):
                request.tools = request.tools + [admin_tool]
            return request

    # Model will try to call admin_tool when admin
    model = FakeToolCallingModel(
        tool_calls=[[{"args": {"input": "test"}, "id": "1", "name": "admin_tool"}], []]
    )

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[ConditionalToolMiddleware()],
    )

    # Test with admin user
    result = agent.invoke({"messages": [HumanMessage("Use admin tool")], "is_admin": True})
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "admin_tool"

    # Test with non-admin user (should not have admin tool available)
    model_no_admin = FakeToolCallingModel(tool_calls=[[], []])  # No tool calls
    agent_no_admin = create_agent(
        model=model_no_admin,
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[ConditionalToolMiddleware()],
    )

    result_no_admin = agent_no_admin.invoke(
        {"messages": [HumanMessage("Hello")], "is_admin": False}
    )
    messages_no_admin = result_no_admin["messages"]
    tool_messages_no_admin = [m for m in messages_no_admin if isinstance(m, ToolMessage)]
    assert len(tool_messages_no_admin) == 0


def test_empty_tools_list_is_valid() -> None:
    """Test that middleware can set an empty tools list."""

    @tool
    def test_tool(input: str) -> str:
        """Test tool."""
        return "test"

    class EmptyToolsMiddleware(AgentMiddleware):
        def modify_model_request(
            self, request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> ModelRequest:
            # Remove all tools
            request.tools = []
            return request

    # Model should not make any tool calls
    model = FakeToolCallingModel(tool_calls=[[], []])  # No tool calls

    agent = create_agent(
        model=model,
        tools=[test_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[EmptyToolsMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("Hello")]})

    assert "messages" in result


def test_tools_preserved_across_multiple_middleware() -> None:
    """Test that tool modifications by one middleware are visible to the next."""
    modification_order: list[list[str]] = []

    @tool
    def tool_a(input: str) -> str:
        """Tool A."""
        return "A"

    @tool
    def tool_b(input: str) -> str:
        """Tool B."""
        return "B"

    @tool
    def tool_c(input: str) -> str:
        """Tool C."""
        return "C"

    class FirstMiddleware(AgentMiddleware):
        def modify_model_request(
            self, request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> ModelRequest:
            modification_order.append([t.name for t in request.tools])
            # Remove tool_c
            request.tools = [t for t in request.tools if t.name != "tool_c"]
            return request

    class SecondMiddleware(AgentMiddleware):
        def modify_model_request(
            self, request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> ModelRequest:
            modification_order.append([t.name for t in request.tools])
            # Should not see tool_c here
            assert "tool_c" not in [t.name for t in request.tools]
            # Remove tool_b
            request.tools = [t for t in request.tools if t.name != "tool_b"]
            return request

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
    def middleware_tool(input: str) -> str:
        """Tool provided by middleware."""
        return f"Middleware tool result: {input}"

    class ToolProvidingMiddleware(AgentMiddleware):
        def __init__(self):
            super().__init__()
            self.tools = [middleware_tool]

        def modify_model_request(
            self, request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> ModelRequest:
            # Add middleware tools to the request
            request.tools = request.tools + self.tools
            return request

    # Model will try to call middleware_tool
    model = FakeToolCallingModel(
        tool_calls=[[{"args": {"input": "test"}, "id": "1", "name": "middleware_tool"}], []]
    )

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[ToolProvidingMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("Use middleware tool")]})

    # Verify that the middleware tool was executed successfully
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "middleware_tool"
    assert "middleware" in tool_messages[0].content.lower()
