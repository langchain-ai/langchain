"""Unit tests for the @on_tool_call decorator."""

from collections.abc import Generator

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    on_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from tests.unit_tests.agents.test_middleware_agent import FakeToolCallingModel


@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool
def calculator_tool(expression: str) -> str:
    """Calculate an expression."""
    return f"Result: {expression}"


@tool
def failing_tool(input: str) -> str:
    """Tool that always fails."""
    msg = f"Failed: {input}"
    raise ValueError(msg)


class TestOnToolCallDecorator:
    """Test the @on_tool_call decorator for creating middleware."""

    def test_basic_decorator_usage(self) -> None:
        """Test basic decorator usage without parameters."""

        @on_tool_call
        def passthrough_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            response = yield request

        # Should return an AgentMiddleware instance
        assert isinstance(passthrough_middleware, AgentMiddleware)

        # Should work in agent
        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(
                        name="search_tool",
                        args={"query": "test"},
                        id="call_1",
                        type="tool_call",
                    )
                ],
                [],  # Empty to signal agent should stop
            ],
            tool_style="openai",
        )
        agent = create_agent(
            model=model,
            tools=[search_tool],
            middleware=[passthrough_middleware],
            checkpointer=InMemorySaver(),
        )

        result = agent.invoke(
            {"messages": [HumanMessage("Search for test")]},
            {"configurable": {"thread_id": "test"}},
        )

        # Should have human message, AI message with tool call, tool message, and final AI message
        assert len(result["messages"]) >= 3
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert "Results for: test" in tool_messages[0].content

    def test_decorator_with_custom_name(self) -> None:
        """Test decorator with custom middleware name."""

        @on_tool_call(name="CustomToolMiddleware")
        def my_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            response = yield request

        assert isinstance(my_middleware, AgentMiddleware)
        assert my_middleware.__class__.__name__ == "CustomToolMiddleware"

    def test_decorator_logging(self) -> None:
        """Test decorator for logging tool calls."""
        call_log = []

        @on_tool_call
        def logging_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            call_log.append(f"before_{request.tool.name}")
            response = yield request
            call_log.append(f"after_{request.tool.name}")

        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(
                        name="search_tool",
                        args={"query": "test"},
                        id="call_1",
                        type="tool_call",
                    )
                ],
                [],  # Empty to signal agent should stop
            ],
            tool_style="openai",
        )
        agent = create_agent(
            model=model,
            tools=[search_tool],
            middleware=[logging_middleware],
            checkpointer=InMemorySaver(),
        )

        agent.invoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test"}},
        )

        assert call_log == ["before_search_tool", "after_search_tool"]

    def test_decorator_modifying_args(self) -> None:
        """Test decorator modifying tool arguments."""

        @on_tool_call
        def modify_args_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            # Modify the query argument
            request.tool_call["args"]["query"] = "modified query"
            response = yield request

        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(
                        name="search_tool",
                        args={"query": "original"},
                        id="call_1",
                        type="tool_call",
                    )
                ],
                [],  # Empty to signal agent should stop
            ],
            tool_style="openai",
        )
        agent = create_agent(
            model=model,
            tools=[search_tool],
            middleware=[modify_args_middleware],
            checkpointer=InMemorySaver(),
        )

        result = agent.invoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test"}},
        )

        # Tool should have been called with modified args
        tool_message = result["messages"][2]
        assert "Results for: modified query" in tool_message.content

    def test_decorator_response_inspection(self) -> None:
        """Test decorator inspecting tool responses."""
        inspected_values = []

        @on_tool_call
        def inspect_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            response = yield request
            inspected_values.append(response.content)

        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(
                        name="search_tool",
                        args={"query": "test"},
                        id="call_1",
                        type="tool_call",
                    )
                ],
                [],  # Empty to signal agent should stop
            ],
            tool_style="openai",
        )
        agent = create_agent(
            model=model,
            tools=[search_tool],
            middleware=[inspect_middleware],
            checkpointer=InMemorySaver(),
        )

        result = agent.invoke(
            {"messages": [HumanMessage("Test")]},
            {"configurable": {"thread_id": "test"}},
        )

        # Should have inspected the response
        assert len(inspected_values) == 1
        assert "Results for: test" in inspected_values[0]

    def test_decorator_with_state_access(self) -> None:
        """Test decorator accessing agent state."""
        state_values = []

        @on_tool_call
        def log_state(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            state_values.append(len(state.get("messages", [])))
            response = yield request

        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(
                        name="search_tool",
                        args={"query": "test"},
                        id="call_1",
                        type="tool_call",
                    )
                ],
                [],  # Empty to signal agent should stop
            ],
            tool_style="openai",
        )
        agent = create_agent(
            model=model,
            tools=[search_tool],
            middleware=[log_state],
            checkpointer=InMemorySaver(),
        )

        agent.invoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test"}},
        )

        # State should have been logged
        assert len(state_values) == 1
        assert state_values[0] == 2  # Human message + AI message with tool call

    def test_multiple_decorated_middleware(self) -> None:
        """Test composition of multiple decorated middleware."""
        execution_order = []

        @on_tool_call
        def outer_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            execution_order.append("outer-before")
            response = yield request
            execution_order.append("outer-after")

        @on_tool_call
        def inner_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            execution_order.append("inner-before")
            response = yield request
            execution_order.append("inner-after")

        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(
                        name="search_tool",
                        args={"query": "test"},
                        id="call_1",
                        type="tool_call",
                    )
                ],
                [],  # Empty to signal agent should stop
            ],
            tool_style="openai",
        )
        agent = create_agent(
            model=model,
            tools=[search_tool],
            middleware=[outer_middleware, inner_middleware],
            checkpointer=InMemorySaver(),
        )

        agent.invoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test"}},
        )

        assert execution_order == [
            "outer-before",
            "inner-before",
            "inner-after",
            "outer-after",
        ]

    def test_decorator_with_custom_state_schema(self) -> None:
        """Test decorator with custom state schema."""
        from typing_extensions import TypedDict

        class CustomState(TypedDict):
            messages: list
            custom_field: str

        @on_tool_call(state_schema=CustomState)
        def middleware_with_schema(
            request: ToolCallRequest, state, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            response = yield request

        assert isinstance(middleware_with_schema, AgentMiddleware)
        assert middleware_with_schema.state_schema == CustomState

    def test_decorator_with_tools_parameter(self) -> None:
        """Test decorator with tools parameter."""

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Result: {query}"

        @on_tool_call(tools=[test_tool])
        def middleware_with_tools(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            response = yield request

        assert isinstance(middleware_with_tools, AgentMiddleware)
        assert len(middleware_with_tools.tools) == 1
        assert middleware_with_tools.tools[0].name == "test_tool"

    def test_decorator_parentheses_optional(self) -> None:
        """Test that decorator works both with and without parentheses."""

        # Without parentheses
        @on_tool_call
        def middleware_no_parens(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            response = yield request

        # With parentheses
        @on_tool_call()
        def middleware_with_parens(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            response = yield request

        assert isinstance(middleware_no_parens, AgentMiddleware)
        assert isinstance(middleware_with_parens, AgentMiddleware)

    def test_decorator_preserves_function_name(self) -> None:
        """Test that decorator uses function name for class name."""

        @on_tool_call
        def my_custom_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            response = yield request

        assert my_custom_middleware.__class__.__name__ == "my_custom_middleware"

    def test_decorator_mixed_with_class_middleware(self) -> None:
        """Test decorated middleware mixed with class-based middleware."""
        execution_order = []

        @on_tool_call
        def decorated_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            execution_order.append("decorated-before")
            response = yield request
            execution_order.append("decorated-after")

        class ClassMiddleware(AgentMiddleware):
            def on_tool_call(
                self, request: ToolCallRequest, state, runtime
            ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
                execution_order.append("class-before")
                response = yield request
                execution_order.append("class-after")

        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(
                        name="search_tool",
                        args={"query": "test"},
                        id="call_1",
                        type="tool_call",
                    )
                ],
                [],  # Empty to signal agent should stop
            ],
            tool_style="openai",
        )
        agent = create_agent(
            model=model,
            tools=[search_tool],
            middleware=[decorated_middleware, ClassMiddleware()],
            checkpointer=InMemorySaver(),
        )

        agent.invoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test"}},
        )

        assert execution_order == [
            "decorated-before",
            "class-before",
            "class-after",
            "decorated-after",
        ]

    def test_decorator_short_circuit_with_cached_result(self) -> None:
        """Test decorator short-circuiting with cached result."""
        cache = {}

        @on_tool_call
        def caching_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            cache_key = f"{request.tool_call['name']}:{request.tool_call['args']}"
            if cache_key in cache:
                # Short-circuit with cached result
                yield cache[cache_key]
            else:
                # Execute tool and cache result
                response = yield request
                cache[cache_key] = response

        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(
                        name="search_tool",
                        args={"query": "test"},
                        id="call_1",
                        type="tool_call",
                    )
                ],
                [
                    ToolCall(
                        name="search_tool",
                        args={"query": "test"},
                        id="call_2",
                        type="tool_call",
                    )
                ],
                [],  # Empty to signal agent should stop
            ],
            tool_style="openai",
        )
        agent = create_agent(
            model=model,
            tools=[search_tool],
            middleware=[caching_middleware],
            checkpointer=InMemorySaver(),
        )

        # First call - should execute tool
        result1 = agent.invoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test1"}},
        )

        # Cache should be populated
        assert len(cache) == 1

        # Second call - should use cache
        result2 = agent.invoke(
            {"messages": [HumanMessage("Search again")]},
            {"configurable": {"thread_id": "test2"}},
        )

        # Both results should have tool messages with same content
        assert "Results for: test" in result1["messages"][2].content
        assert "Results for: test" in result2["messages"][2].content

    async def test_decorator_with_async_agent(self) -> None:
        """Test that decorated middleware works with async agent invocation."""
        call_log = []

        @on_tool_call
        def logging_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            call_log.append("before")
            response = yield request
            call_log.append("after")

        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(
                        name="search_tool",
                        args={"query": "test"},
                        id="call_1",
                        type="tool_call",
                    )
                ],
                [],  # Empty to signal agent should stop
            ],
            tool_style="openai",
        )
        agent = create_agent(
            model=model,
            tools=[search_tool],
            middleware=[logging_middleware],
            checkpointer=InMemorySaver(),
        )

        result = await agent.ainvoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test"}},
        )

        assert call_log == ["before", "after"]
        assert "Results for: test" in result["messages"][2].content

    def test_decorator_multiple_tools_called(self) -> None:
        """Test decorator handling multiple tool calls in one turn."""
        call_log = []

        @on_tool_call
        def logging_middleware(
            request: ToolCallRequest, state: AgentState, runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            call_log.append(request.tool.name)
            response = yield request

        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(
                        name="search_tool",
                        args={"query": "test"},
                        id="call_1",
                        type="tool_call",
                    ),
                    ToolCall(
                        name="calculator_tool",
                        args={"expression": "2+2"},
                        id="call_2",
                        type="tool_call",
                    ),
                ],
                [],  # Empty to signal agent should stop
            ],
            tool_style="openai",
        )
        agent = create_agent(
            model=model,
            tools=[search_tool, calculator_tool],
            middleware=[logging_middleware],
            checkpointer=InMemorySaver(),
        )

        result = agent.invoke(
            {"messages": [HumanMessage("Search and calculate")]},
            {"configurable": {"thread_id": "test"}},
        )

        # Both tools should have been called
        assert "search_tool" in call_log
        assert "calculator_tool" in call_log
        assert len(result["messages"]) >= 4  # human, ai, tool1, tool2, (final ai)
