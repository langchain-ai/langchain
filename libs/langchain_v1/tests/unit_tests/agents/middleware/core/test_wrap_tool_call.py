"""Tests for wrap_tool_call decorator functionality.

These tests verify the decorator-based approach for wrapping tool calls,
focusing on the handler pattern (not generators).
"""

from collections.abc import Callable

from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import wrap_tool_call
from langchain.agents.middleware.types import ToolCallRequest
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool
def calculator(expression: str) -> str:
    """Calculate an expression."""
    return f"Calculated: {expression}"


@tool
def failing_tool(input: str) -> str:
    """Tool that always fails."""
    msg = f"Failed: {input}"
    raise ValueError(msg)


def test_wrap_tool_call_basic_passthrough() -> None:
    """Test basic passthrough with wrap_tool_call decorator."""
    call_log = []

    @wrap_tool_call
    def passthrough(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("called")
        return handler(request)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[passthrough],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    assert len(call_log) == 1
    assert call_log[0] == "called"
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Results for: test" in tool_messages[0].content


def test_wrap_tool_call_logging() -> None:
    """Test logging tool call execution with wrap_tool_call decorator."""
    call_log = []

    @wrap_tool_call
    def logging_middleware(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append(f"before_{request.tool.name}")
        response = handler(request)
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
        middleware=[logging_middleware],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    assert call_log == ["before_search", "after_search"]
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_wrap_tool_call_modify_args() -> None:
    """Test modifying tool arguments with wrap_tool_call decorator."""

    @wrap_tool_call
    def modify_args(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        # Modify the query argument before execution
        if request.tool.name == "search":
            request.tool_call["args"]["query"] = "modified query"
        return handler(request)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "original"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[modify_args],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "modified query" in tool_messages[0].content


def test_wrap_tool_call_access_state() -> None:
    """Test accessing agent state from wrap_tool_call decorator."""
    state_data = []

    @wrap_tool_call
    def access_state(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        # Access state from request
        if request.state is not None:
            messages = request.state.get("messages", [])
            state_data.append(len(messages))
        return handler(request)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[access_state],
        checkpointer=InMemorySaver(),
    )

    agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Middleware should have accessed state
    assert len(state_data) >= 1
    assert state_data[0] > 0  # Should have at least the initial message


def test_wrap_tool_call_access_runtime() -> None:
    """Test accessing runtime from wrap_tool_call decorator."""
    runtime_data = []

    @wrap_tool_call
    def access_runtime(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        # Access runtime from request
        if request.runtime is not None:
            # Runtime object is available (has context, store, stream_writer, previous)
            runtime_data.append(type(request.runtime).__name__)
        return handler(request)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[access_runtime],
        checkpointer=InMemorySaver(),
    )

    agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test_thread"}},
    )

    # Middleware should have accessed runtime
    assert len(runtime_data) >= 1
    assert runtime_data[0] == "ToolRuntime"


def test_wrap_tool_call_retry_on_error() -> None:
    """Test retry logic with wrap_tool_call decorator on failing tool."""
    attempt_counts = []

    @wrap_tool_call
    def retry_middleware(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            attempt_counts.append(attempt)
            try:
                return handler(request)
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    # Return error message instead of raising
                    return ToolMessage(
                        content=f"Error after {max_retries} attempts: {str(last_error)}",
                        tool_call_id=request.tool_call["id"],
                        name=request.tool_call["name"],
                        status="error",
                    )
                # Continue to retry
        # This line should never be reached due to return above
        return ToolMessage(
            content=f"Unexpected error: {str(last_error)}",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"input": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[retry_middleware],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Should attempt 3 times before giving up
    assert len(attempt_counts) == 3
    assert attempt_counts == [0, 1, 2]
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Error after 3 attempts" in tool_messages[0].content


def test_wrap_tool_call_short_circuit() -> None:
    """Test short-circuiting tool execution with wrap_tool_call decorator."""
    handler_called = []

    @wrap_tool_call
    def short_circuit(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        # Don't call handler, return custom response directly
        handler_called.append(False)
        return ToolMessage(
            content="short_circuit_result",
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
        middleware=[short_circuit],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Handler was not called
    assert len(handler_called) == 1
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "short_circuit_result" in tool_messages[0].content


def test_wrap_tool_call_response_modification() -> None:
    """Test modifying tool response with wrap_tool_call decorator."""

    @wrap_tool_call
    def modify_response(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        response = handler(request)

        # Modify the response
        if isinstance(response, ToolMessage):
            modified = ToolMessage(
                content=f"MODIFIED: {response.content}",
                tool_call_id=response.tool_call_id,
                name=response.name,
            )
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
        middleware=[modify_response],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "MODIFIED: Results for: test" in tool_messages[0].content


def test_wrap_tool_call_multiple_middleware_composition() -> None:
    """Test multiple wrap_tool_call middleware compose correctly."""
    call_log = []

    @wrap_tool_call
    def outer_middleware(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("outer_before")
        response = handler(request)
        call_log.append("outer_after")
        return response

    @wrap_tool_call
    def inner_middleware(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("inner_before")
        response = handler(request)
        call_log.append("inner_after")
        return response

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    # First middleware in list is outermost
    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[outer_middleware, inner_middleware],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify correct composition order
    assert call_log == ["outer_before", "inner_before", "inner_after", "outer_after"]
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_wrap_tool_call_multiple_tools() -> None:
    """Test wrap_tool_call handles multiple tool calls correctly."""
    call_log = []

    @wrap_tool_call
    def log_tool_calls(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append(request.tool.name)
        return handler(request)

    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "test"}, id="1"),
                ToolCall(name="calculator", args={"expression": "1+1"}, id="2"),
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, calculator],
        middleware=[log_tool_calls],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use tools")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Both tools should be logged
    assert "search" in call_log
    assert "calculator" in call_log
    assert len(call_log) == 2

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2


def test_wrap_tool_call_with_custom_name() -> None:
    """Test wrap_tool_call decorator with custom middleware name."""

    @wrap_tool_call(name="CustomToolWrapper")
    def my_wrapper(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        return handler(request)

    # Verify custom name was applied
    assert my_wrapper.__class__.__name__ == "CustomToolWrapper"


def test_wrap_tool_call_with_tools_parameter() -> None:
    """Test wrap_tool_call decorator with tools parameter."""

    @tool
    def extra_tool(input: str) -> str:
        """Extra tool registered with middleware."""
        return f"Extra: {input}"

    @wrap_tool_call(tools=[extra_tool])
    def wrapper_with_tools(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        return handler(request)

    # Verify tools were registered
    assert wrapper_with_tools.tools == [extra_tool]


def test_wrap_tool_call_three_levels_composition() -> None:
    """Test composition with three wrap_tool_call middleware levels."""
    call_log = []

    @wrap_tool_call(name="OuterWrapper")
    def outer(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("outer_before")
        response = handler(request)
        call_log.append("outer_after")
        return response

    @wrap_tool_call(name="MiddleWrapper")
    def middle(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("middle_before")
        response = handler(request)
        call_log.append("middle_after")
        return response

    @wrap_tool_call(name="InnerWrapper")
    def inner(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("inner_before")
        response = handler(request)
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
        middleware=[outer, middle, inner],
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


def test_wrap_tool_call_outer_intercepts_inner() -> None:
    """Test composition where outer middleware intercepts inner response."""
    call_log = []

    @wrap_tool_call(name="InterceptingOuter")
    def intercepting_outer(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("outer_before")
        response = handler(request)
        call_log.append("outer_after")

        # Return modified message
        return ToolMessage(
            content="Outer intercepted",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    @wrap_tool_call(name="InnerWrapper")
    def inner(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("inner_called")
        response = handler(request)
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
        middleware=[intercepting_outer, inner],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Both should be called, outer intercepts the response
    assert call_log == [
        "outer_before",
        "inner_called",
        "inner_got_response",
        "outer_after",
    ]

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Outer intercepted" in tool_messages[0].content


def test_wrap_tool_call_inner_short_circuits() -> None:
    """Test composition when inner middleware short-circuits."""
    call_log = []

    @wrap_tool_call(name="OuterWrapper")
    def outer(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("outer_before")
        response = handler(request)
        call_log.append("outer_after")

        # Wrap inner's response
        if isinstance(response, ToolMessage):
            return ToolMessage(
                content=f"outer_wrapped: {response.content}",
                tool_call_id=response.tool_call_id,
                name=response.name,
            )
        return response

    @wrap_tool_call(name="InnerShortCircuit")
    def inner_short_circuit(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("inner_short_circuit")
        # Don't call handler, return custom response
        return ToolMessage(
            content="inner_result",
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
        middleware=[outer, inner_short_circuit],
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
    assert "outer_wrapped: inner_result" in tool_messages[0].content


def test_wrap_tool_call_mixed_passthrough_and_intercepting() -> None:
    """Test composition with mix of pass-through and intercepting handlers."""
    call_log = []

    @wrap_tool_call(name="FirstPassthrough")
    def first_passthrough(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("first_before")
        response = handler(request)
        call_log.append("first_after")
        return response

    @wrap_tool_call(name="SecondIntercepting")
    def second_intercepting(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("second_intercept")
        # Call handler but ignore result
        _ = handler(request)
        # Return custom result
        return ToolMessage(
            content="intercepted_result",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    @wrap_tool_call(name="ThirdPassthrough")
    def third_passthrough(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        call_log.append("third_called")
        response = handler(request)
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
        middleware=[first_passthrough, second_intercepting, third_passthrough],
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


def test_wrap_tool_call_uses_function_name_as_default() -> None:
    """Test that wrap_tool_call uses function name as default middleware name."""

    @wrap_tool_call
    def my_custom_wrapper(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        return handler(request)

    # Verify that function name is used as middleware class name
    assert my_custom_wrapper.__class__.__name__ == "my_custom_wrapper"


def test_wrap_tool_call_caching_pattern() -> None:
    """Test caching pattern with wrap_tool_call decorator."""
    cache = {}
    handler_calls = []

    @wrap_tool_call
    def with_cache(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        # Create cache key from tool name and args
        cache_key = (request.tool.name, str(request.tool_call["args"]))

        # Check cache
        if cache_key in cache:
            return ToolMessage(
                content=cache[cache_key],
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )

        # Execute tool and cache result
        handler_calls.append("executed")
        response = handler(request)

        if isinstance(response, ToolMessage):
            cache[cache_key] = response.content

        return response

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [ToolCall(name="search", args={"query": "test"}, id="2")],  # Same query
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[with_cache],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search twice")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Handler should only be called once (second call uses cache)
    assert len(handler_calls) == 1

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    # Both tool calls should have messages
    assert len(tool_messages) >= 1


def test_wrap_tool_call_monitoring_pattern() -> None:
    """Test monitoring pattern with wrap_tool_call decorator."""
    metrics = []

    @wrap_tool_call
    def monitor_execution(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        import time

        start_time = time.time()
        response = handler(request)
        execution_time = time.time() - start_time

        metrics.append(
            {
                "tool": request.tool.name,
                "execution_time": execution_time,
                "success": isinstance(response, ToolMessage)
                and not response.content.startswith("Error:"),
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
        middleware=[monitor_execution],
        checkpointer=InMemorySaver(),
    )

    agent.invoke(
        {"messages": [HumanMessage("Search")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Metrics should be collected
    assert len(metrics) == 1
    assert metrics[0]["tool"] == "search"
    assert metrics[0]["success"] is True
    assert metrics[0]["execution_time"] >= 0
