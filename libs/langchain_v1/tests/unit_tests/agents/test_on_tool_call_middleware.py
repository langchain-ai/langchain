"""Unit tests for wrap_tool_call middleware integration."""

from collections.abc import Callable

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from tests.unit_tests.agents.test_middleware_agent import FakeToolCallingModel


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool
def calculator(expression: str) -> str:
    """Calculate an expression."""
    return f"Result: {expression}"


@tool
def failing_tool(input: str) -> str:
    """Tool that always fails."""
    msg = f"Failed: {input}"
    raise ValueError(msg)


def test_simple_logging_middleware() -> None:
    """Test middleware that logs tool calls."""
    call_log = []

    class LoggingMiddleware(AgentMiddleware):
        """Middleware that logs tool calls."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append(f"before_{request.tool.name}")
            response = handler(request.tool_call)
            call_log.append(f"after_{request.tool.name}")
            return response

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[LoggingMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    assert len(call_log) == 2
    assert call_log[0] == "before_search"
    assert call_log[1] == "after_search"
    assert len(result["messages"]) > 0


def test_request_modification_middleware() -> None:
    """Test middleware that modifies tool call arguments."""

    class ModifyArgsMiddleware(AgentMiddleware):
        """Middleware that modifies tool arguments."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            # Add prefix to query
            if request.tool.name == "search":
                original_query = request.tool_call["args"]["query"]
                request.tool_call["args"]["query"] = f"modified: {original_query}"
            return handler(request.tool_call)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[ModifyArgsMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "modified: test" in tool_messages[0].content


def test_response_inspection_middleware() -> None:
    """Test middleware that inspects tool responses."""
    inspected_responses = []

    class ResponseInspectionMiddleware(AgentMiddleware):
        """Middleware that inspects responses."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            response = handler(request.tool_call)

            # Record response details
            if isinstance(response, ToolMessage):
                inspected_responses.append(
                    {
                        "tool_name": request.tool.name,
                        "content": response.content,
                    }
                )
            return response

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[ResponseInspectionMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    # Middleware should have inspected the response
    assert len(inspected_responses) == 1
    assert inspected_responses[0]["tool_name"] == "search"


def test_conditional_retry_middleware() -> None:
    """Test middleware that retries tool calls based on response content."""
    call_count = 0

    class ConditionalRetryMiddleware(AgentMiddleware):
        """Middleware that retries based on response content."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            nonlocal call_count
            max_retries = 2

            for attempt in range(max_retries):
                response = handler(request.tool_call)
                call_count += 1

                # Check if we should retry based on content
                if (
                    isinstance(response, ToolMessage)
                    and "retry_marker" in response.content
                    and attempt < max_retries - 1
                ):
                    # Continue to retry
                    continue

                # Return on success or final attempt
                return response

            return response

    # Use search tool which always succeeds - we'll modify request to test retry logic
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[ConditionalRetryMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Middleware should have been called at least once
    assert call_count >= 1
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_multiple_middleware_composition() -> None:
    """Test that multiple middleware compose correctly."""
    call_log = []

    class OuterMiddleware(AgentMiddleware):
        """Outer middleware."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("outer_before")
            response = handler(request.tool_call)
            call_log.append("outer_after")
            return response

    class InnerMiddleware(AgentMiddleware):
        """Inner middleware."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("inner_before")
            response = handler(request.tool_call)
            call_log.append("inner_after")
            return response

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    # First middleware is outermost
    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[OuterMiddleware(), InnerMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify correct composition order
    assert call_log == ["outer_before", "inner_before", "inner_after", "outer_after"]
    assert len(result["messages"]) > 0


def test_middleware_with_multiple_tool_calls() -> None:
    """Test middleware handles multiple tool calls correctly."""
    call_log = []

    class LoggingMiddleware(AgentMiddleware):
        """Middleware that logs tool calls."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append(request.tool.name)
            return handler(request.tool_call)

    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "test1"}, id="1"),
                ToolCall(name="calculator", args={"expression": "1+1"}, id="2"),
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, calculator],
        middleware=[LoggingMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use tools")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Each tool call should be logged
    assert "search" in call_log
    assert "calculator" in call_log
    assert len(call_log) == 2

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2


def test_middleware_access_to_state() -> None:
    """Test middleware can access agent state."""
    state_seen = []

    class StateInspectionMiddleware(AgentMiddleware):
        """Middleware that inspects state."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            # Record state - state is now in request.state
            state = request.state
            if state is not None:
                if isinstance(state, dict) and "messages" in state:
                    state_seen.append(("dict", len(state["messages"])))
                elif isinstance(state, list):
                    state_seen.append(("list", len(state)))
                else:
                    state_seen.append(("other", type(state).__name__))
            return handler(request.tool_call)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[StateInspectionMiddleware()],
        checkpointer=InMemorySaver(),
    )

    agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Middleware should have seen state (state is passed to on_tool_call)
    assert len(state_seen) >= 1


def test_middleware_without_on_tool_call() -> None:
    """Test that middleware without on_tool_call hook works normally."""

    class NoOpMiddleware(AgentMiddleware):
        """Middleware without on_tool_call."""

        def before_model(self, state, runtime):
            """Just a dummy hook."""
            return None

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[NoOpMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Should work normally
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_generator_composition_immediate_outer_return() -> None:
    """Test composition when outer handler intercepts response."""
    call_log = []

    class ImmediateReturnMiddleware(AgentMiddleware):
        """Outer middleware that intercepts and modifies response."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("outer_yield")
            # Call handler, receive response from inner
            response = handler(request.tool_call)
            call_log.append("outer_got_response")
            # Return modified message
            modified = ToolMessage(
                content="Outer intercepted",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )
            return modified

    class InnerMiddleware(AgentMiddleware):
        """Inner middleware."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("inner_called")
            response = handler(request.tool_call)
            call_log.append("inner_got_response")
            return response

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[ImmediateReturnMiddleware(), InnerMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Both should be called, outer intercepts the response
    assert call_log == ["outer_yield", "inner_called", "inner_got_response", "outer_got_response"]

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Outer intercepted" in tool_messages[0].content


def test_generator_composition_short_circuit() -> None:
    """Test composition when inner handler short-circuits."""
    call_log = []

    class OuterMiddleware(AgentMiddleware):
        """Outer middleware."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("outer_before")
            response = handler(request.tool_call)
            call_log.append("outer_after")
            # Modify response from inner
            if isinstance(response, ToolMessage):
                modified = ToolMessage(
                    content=f"outer_wrapped: {response.content}",
                    tool_call_id=response.tool_call_id,
                    name=response.name,
                )
                return modified
            return response

    class InnerShortCircuitMiddleware(AgentMiddleware):
        """Inner middleware that short-circuits without calling actual tool."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("inner_short_circuit")
            # Don't call handler, return custom response directly
            return ToolMessage(
                content="inner_short_circuit_result",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[OuterMiddleware(), InnerShortCircuitMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify order: outer_before -> inner short circuits -> outer_after
    assert call_log == ["outer_before", "inner_short_circuit", "outer_after"]

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "outer_wrapped: inner_short_circuit_result" in tool_messages[0].content


def test_generator_composition_nested_retries() -> None:
    """Test composition when both outer and inner handlers retry."""
    call_log = []

    class OuterRetryMiddleware(AgentMiddleware):
        """Outer middleware with retry logic."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            for outer_attempt in range(2):
                call_log.append(f"outer_{outer_attempt}")
                response = handler(request.tool_call)

                if isinstance(response, ToolMessage) and response.content == "inner_final_failure":
                    # Inner failed, retry once
                    continue
                return response
            return response

    class InnerRetryMiddleware(AgentMiddleware):
        """Inner middleware with retry logic."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            for inner_attempt in range(2):
                call_log.append(f"inner_{inner_attempt}")
                response = handler(request.tool_call)

                # Check for error in tool result
                if isinstance(response, ToolMessage):
                    if inner_attempt == 0 and "Results for:" in response.content:
                        # First attempt succeeded, but let's pretend it's a soft failure
                        # to test inner retry
                        continue
                    return response

            # Inner exhausted retries
            return ToolMessage(
                content="inner_final_failure",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[OuterRetryMiddleware(), InnerRetryMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify nested retry pattern
    assert "outer_0" in call_log
    assert "inner_0" in call_log
    assert "inner_1" in call_log

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_generator_composition_three_levels() -> None:
    """Test composition with three middleware levels."""
    call_log = []

    class OuterMiddleware(AgentMiddleware):
        """Outermost middleware."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("outer_before")
            response = handler(request.tool_call)
            call_log.append("outer_after")
            return response

    class MiddleMiddleware(AgentMiddleware):
        """Middle middleware."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("middle_before")
            response = handler(request.tool_call)
            call_log.append("middle_after")
            return response

    class InnerMiddleware(AgentMiddleware):
        """Innermost middleware."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("inner_before")
            response = handler(request.tool_call)
            call_log.append("inner_after")
            return response

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[OuterMiddleware(), MiddleMiddleware(), InnerMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify correct nesting order
    assert call_log == [
        "outer_before",
        "middle_before",
        "inner_before",
        "inner_after",
        "middle_after",
        "outer_after",
    ]

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_generator_composition_return_value_extraction() -> None:
    """Test that handler can modify and return response."""
    final_content = []

    class ModifyingMiddleware(AgentMiddleware):
        """Middleware that modifies the final result."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            response = handler(request.tool_call)

            # Return a modified response
            if isinstance(response, ToolMessage):
                modified = ToolMessage(
                    content=f"modified: {response.content}",
                    tool_call_id=response.tool_call_id,
                    name=response.name,
                )
                final_content.append(modified.content)
                return modified
            return response

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[ModifyingMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    # Verify the returned value was properly extracted
    assert "modified:" in tool_messages[0].content
    assert len(final_content) == 1
    assert "modified:" in final_content[0]


def test_generator_composition_with_mixed_passthrough_and_intercepting() -> None:
    """Test composition with mix of pass-through and intercepting handlers."""
    call_log = []

    class FirstPassthroughMiddleware(AgentMiddleware):
        """First middleware that passes through."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("first_before")
            response = handler(request.tool_call)
            call_log.append("first_after")
            return response

    class SecondInterceptingMiddleware(AgentMiddleware):
        """Second middleware that intercepts and returns custom result."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("second_intercept")
            # Call handler but ignore the result
            _ = handler(request.tool_call)
            # Return custom result
            return ToolMessage(
                content="intercepted_result",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )

    class ThirdPassthroughMiddleware(AgentMiddleware):
        """Third middleware that passes through."""

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCall], ToolMessage | Command],
        ) -> ToolMessage | Command:
            call_log.append("third_called")
            response = handler(request.tool_call)
            call_log.append("third_after")
            return response

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[
            FirstPassthroughMiddleware(),
            SecondInterceptingMiddleware(),
            ThirdPassthroughMiddleware(),
        ],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # All middleware are called, second intercepts and returns custom result
    assert call_log == [
        "first_before",
        "second_intercept",
        "third_called",
        "third_after",
        "first_after",
    ]

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "intercepted_result" in tool_messages[0].content
