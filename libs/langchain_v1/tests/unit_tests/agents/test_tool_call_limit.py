"""Unit tests for ToolCallLimitMiddleware."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.factory import create_agent
from langchain.agents.middleware.tool_call_limit import (
    ToolCallLimitExceededError,
    ToolCallLimitMiddleware,
)
from tests.unit_tests.agents.test_middleware_agent import FakeToolCallingModel


def test_middleware_initialization_validation():
    """Test that middleware initialization validates parameters correctly."""
    # Test that at least one limit must be specified
    with pytest.raises(ValueError, match="At least one limit must be specified"):
        ToolCallLimitMiddleware()

    # Test invalid exit behavior
    with pytest.raises(ValueError, match="Invalid exit_behavior"):
        ToolCallLimitMiddleware(thread_limit=5, exit_behavior="invalid")  # type: ignore[arg-type]

    # Test valid initialization
    middleware = ToolCallLimitMiddleware(thread_limit=5, run_limit=3)
    assert middleware.thread_limit == 5
    assert middleware.run_limit == 3
    assert middleware.exit_behavior == "end"
    assert middleware.tool_name is None

    # Test with tool name
    middleware = ToolCallLimitMiddleware(tool_name="search", thread_limit=5)
    assert middleware.tool_name == "search"
    assert middleware.thread_limit == 5
    assert middleware.run_limit is None


def test_middleware_name_property():
    """Test that the name property includes tool name when specified."""
    # Test without tool name
    middleware = ToolCallLimitMiddleware(thread_limit=5)
    assert middleware.name == "ToolCallLimitMiddleware"

    # Test with tool name
    middleware = ToolCallLimitMiddleware(tool_name="search", thread_limit=5)
    assert middleware.name == "ToolCallLimitMiddleware[search]"

    # Test multiple instances with different tool names have unique names
    middleware1 = ToolCallLimitMiddleware(tool_name="search", thread_limit=5)
    middleware2 = ToolCallLimitMiddleware(tool_name="calculator", thread_limit=3)
    assert middleware1.name != middleware2.name
    assert middleware1.name == "ToolCallLimitMiddleware[search]"
    assert middleware2.name == "ToolCallLimitMiddleware[calculator]"


def test_middleware_unit_functionality():
    """Test that the middleware works as expected in isolation."""
    # Test with end behavior - global tool limit, allow_other_tools=False for old behavior
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2, allow_other_tools=False)

    # Mock runtime (not used in current implementation)
    runtime = None

    # Test when limits are not exceeded
    state = {
        "messages": [
            HumanMessage("Question"),
            AIMessage(
                "Response",
                tool_calls=[{"name": "search", "args": {}, "id": "1"}],
            ),
        ],
        "thread_tool_call_count": {},
        "run_tool_call_count": {},
    }
    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is None

    # Test when thread limit is exceeded
    state = {
        "messages": [
            HumanMessage("Question 1"),
            AIMessage(
                "Response 1",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "calculator", "args": {}, "id": "2"},
                ],
            ),
            ToolMessage("Result 1", tool_call_id="1"),
            ToolMessage("Result 2", tool_call_id="2"),
            HumanMessage("Question 2"),
            AIMessage(
                "Response 2",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "3"},
                    {"name": "calculator", "args": {}, "id": "4"},
                ],
            ),
        ],
        "thread_tool_call_count": {"__all__": 4},  # 4 tool calls total (exceeds limit of 3)
        "run_tool_call_count": {"__all__": 2},  # 2 from second AI message
    }
    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "Thread limit exceeded (4/3 calls)" in result["messages"][0].content

    # Test when run limit is exceeded
    state = {
        "messages": [
            HumanMessage("Question"),
            AIMessage(
                "Response",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "calculator", "args": {}, "id": "2"},
                    {"name": "calculator", "args": {}, "id": "3"},
                ],
            ),
        ],
        "thread_tool_call_count": {"__all__": 3},
        "run_tool_call_count": {"__all__": 3},  # 3 tool calls in current run (exceeds limit of 2)
    }
    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "Run limit exceeded (3/2 calls)" in result["messages"][0].content


def test_middleware_with_specific_tool():
    """Test middleware that limits a specific tool."""
    # Limit only the "search" tool, use allow_other_tools=False for old behavior
    middleware = ToolCallLimitMiddleware(tool_name="search", thread_limit=2, run_limit=1, allow_other_tools=False)

    runtime = None

    # Test with search tool calls - exceeding run limit
    state = {
        "messages": [
            HumanMessage("Question"),
            AIMessage(
                "Response",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "search", "args": {}, "id": "2"},
                    {"name": "calculator", "args": {}, "id": "3"},
                ],
            ),
        ],
        "thread_tool_call_count": {"search": 2},  # 2 search calls
        "run_tool_call_count": {"search": 2},  # 2 search calls in current run (exceeds limit of 1)
    }
    # Run limit for search is 1, should be exceeded
    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert "'search' tool" in result["messages"][0].content
    assert "Run limit exceeded (2/1 calls)" in result["messages"][0].content

    # Test with only calculator calls (should not trigger limit)
    state = {
        "messages": [
            HumanMessage("Question"),
            AIMessage(
                "Response",
                tool_calls=[
                    {"name": "calculator", "args": {}, "id": "1"},
                    {"name": "calculator", "args": {}, "id": "2"},
                ],
            ),
        ],
        "thread_tool_call_count": {},  # 0 search calls
        "run_tool_call_count": {},  # 0 search calls
    }
    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is None


def test_middleware_error_behavior():
    """Test middleware with error exit behavior."""
    middleware = ToolCallLimitMiddleware(thread_limit=2, run_limit=1, exit_behavior="error")

    runtime = None

    # Test exception when thread limit exceeded
    state = {
        "messages": [
            HumanMessage("Q"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
            ToolMessage("Result", tool_call_id="1"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "2"}]),
            ToolMessage("Result", tool_call_id="2"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "3"}]),
        ],
        "thread_tool_call_count": {"__all__": 3},  # 3 tool calls total (exceeds limit of 2)
        "run_tool_call_count": {"__all__": 1},  # 1 tool call in current run
    }
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.before_model(state, runtime)  # type: ignore[arg-type]

    assert "Thread limit exceeded (3/2 calls)" in str(exc_info.value)
    assert exc_info.value.thread_count == 3
    assert exc_info.value.thread_limit == 2

    # Test exception when run limit exceeded
    state = {
        "messages": [
            HumanMessage("Question"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
            ToolMessage("Result", tool_call_id="1"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "2"}]),
        ],
        "thread_tool_call_count": {"__all__": 2},
        "run_tool_call_count": {"__all__": 2},  # 2 tool calls in current run (exceeds limit of 1)
    }
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.before_model(state, runtime)  # type: ignore[arg-type]

    assert "Run limit exceeded (2/1 calls)" in str(exc_info.value)


def test_multiple_middleware_instances():
    """Test using multiple ToolCallLimitMiddleware instances in the same agent."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    @tool
    def calculator(expression: str) -> str:
        """Calculate an expression."""
        return f"Result: {expression}"

    # Create model that makes multiple tool calls
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "test1"}, id="1"),
                ToolCall(name="calculator", args={"expression": "1+1"}, id="2"),
            ],
            [
                ToolCall(name="search", args={"query": "test2"}, id="3"),
                ToolCall(name="calculator", args={"expression": "2+2"}, id="4"),
            ],
            [
                ToolCall(name="search", args={"query": "test3"}, id="5"),
            ],
            [],
        ]
    )

    # Global limit: max 5 tool calls per thread
    # Search-specific limit: max 2 search calls per thread
    global_limiter = ToolCallLimitMiddleware(thread_limit=5)
    search_limiter = ToolCallLimitMiddleware(tool_name="search", thread_limit=2)

    agent = create_agent(
        model=model,
        tools=[search, calculator],
        middleware=[global_limiter, search_limiter],
        checkpointer=InMemorySaver(),
    )

    # First invocation: 2 search calls, 2 calculator calls (4 total)
    # Should succeed - within both limits
    result = agent.invoke(
        {"messages": [HumanMessage("First question")]},
        {"configurable": {"thread_id": "test_thread"}},
    )
    assert len(result["messages"]) > 0

    # Second invocation would add 1 more search call (total 3), exceeding search limit
    # Should hit the search-specific limit
    result = agent.invoke(
        {"messages": [HumanMessage("Second question")]},
        {"configurable": {"thread_id": "test_thread"}},
    )

    # Check that the limit was hit
    last_message = result["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert "'search' tool" in last_message.content
    assert "Thread limit exceeded" in last_message.content


def test_run_limit_with_multiple_human_messages():
    """Test that run limit correctly resets after each HumanMessage."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    # Model makes 2 tool calls, then 1 tool call, then another tool call to exceed limit
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "test1"}, id="1"),
                ToolCall(name="search", args={"query": "test2"}, id="2"),
            ],
            [ToolCall(name="search", args={"query": "test3"}, id="3")],
            [],
        ]
    )

    # Run limit of 2 tool calls per run, allow_other_tools=False to trigger on before_model
    middleware = ToolCallLimitMiddleware(run_limit=2, allow_other_tools=False)

    agent = create_agent(
        model=model, tools=[search], middleware=[middleware], checkpointer=InMemorySaver()
    )

    thread_config = {"configurable": {"thread_id": "test_thread"}}

    # First run: 2 tool calls, then tries to make another call which exceeds the limit
    result = agent.invoke({"messages": [HumanMessage("Question 1")]}, thread_config)
    last_message = result["messages"][-1]
    assert "Run limit exceeded (3/2 calls)" in last_message.content

    # Second run: starts fresh, should also hit run limit after exceeding 2 calls
    result = agent.invoke({"messages": [HumanMessage("Question 2")]}, thread_config)
    last_message = result["messages"][-1]
    # This should also hit the limit because run count resets but then exceeds again
    assert "Run limit exceeded (3/2 calls)" in last_message.content


def test_exception_error_messages():
    """Test that exceptions provide clear error messages."""
    # Test global tool limit
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2, exit_behavior="error")

    state = {
        "messages": [
            HumanMessage("Q"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
            ToolMessage("Result", tool_call_id="1"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "2"}]),
            ToolMessage("Result", tool_call_id="2"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "3"}]),
            ToolMessage("Result", tool_call_id="3"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "4"}]),
        ],
        "thread_tool_call_count": {"__all__": 4},  # 4 tool calls total (exceeds limit of 3)
        "run_tool_call_count": {"__all__": 3},  # 3 tool calls in current run (exceeds limit of 2)
    }

    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.before_model(state, None)  # type: ignore[arg-type]

    error_msg = str(exc_info.value)
    assert "Tool call limit reached" in error_msg
    assert "Thread limit exceeded (4/3 calls)" in error_msg

    # Test specific tool limit
    middleware = ToolCallLimitMiddleware(tool_name="search", thread_limit=2, exit_behavior="error")

    state = {
        "messages": [
            HumanMessage("Q"),
            AIMessage(
                "R",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "calculator", "args": {}, "id": "2"},
                ],
            ),
            ToolMessage("Result", tool_call_id="1"),
            ToolMessage("Result", tool_call_id="2"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "3"}]),
            ToolMessage("Result", tool_call_id="3"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "4"}]),
        ],
        "thread_tool_call_count": {
            "search": 3
        },  # 3 search calls total (exceeds limit of 2)
        "run_tool_call_count": {"search": 2},  # 2 search calls in current run
    }

    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.before_model(state, None)  # type: ignore[arg-type]

    error_msg = str(exc_info.value)
    assert "'search' tool call limit reached" in error_msg
    assert "Thread limit exceeded (3/2 calls)" in error_msg


def test_limit_reached_but_not_exceeded():
    """Test that limits are only triggered when exceeded (>), not when reached (==)."""
    # Use allow_other_tools=False to test the before_model behavior
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2, allow_other_tools=False)

    runtime = None

    # Test when thread limit is reached exactly
    state = {
        "messages": [
            HumanMessage("Q"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
            ToolMessage("Result", tool_call_id="1"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "2"}]),
            ToolMessage("Result", tool_call_id="2"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "3"}]),
        ],
        "thread_tool_call_count": {"__all__": 3},  # Exactly at limit
        "run_tool_call_count": {"__all__": 1},
    }
    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is None  # Should not trigger - limit reached but not exceeded

    # Test when run limit is reached exactly
    state = {
        "messages": [
            HumanMessage("Q"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
            ToolMessage("Result", tool_call_id="1"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "2"}]),
        ],
        "thread_tool_call_count": {"__all__": 2},
        "run_tool_call_count": {"__all__": 2},  # Exactly at limit
    }
    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is None  # Should not trigger - limit reached but not exceeded

    # Test when limit is exceeded by 1
    state = {
        "messages": [
            HumanMessage("Q"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
            ToolMessage("Result", tool_call_id="1"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "2"}]),
            ToolMessage("Result", tool_call_id="2"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "3"}]),
            ToolMessage("Result", tool_call_id="3"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "4"}]),
        ],
        "thread_tool_call_count": {"__all__": 4},  # Exceeds limit of 3
        "run_tool_call_count": {"__all__": 2},
    }
    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None  # Should trigger - limit exceeded
    assert result["jump_to"] == "end"


def test_allow_other_tools_true():
    """Test that allow_other_tools=True allows other tools to continue."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    @tool
    def calculator(expression: str) -> str:
        """Calculate an expression."""
        return f"Result: {expression}"

    # Create model that makes multiple tool calls
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "test1"}, id="1"),
                ToolCall(name="calculator", args={"expression": "1+1"}, id="2"),
            ],
            [
                ToolCall(name="search", args={"query": "test2"}, id="3"),
                ToolCall(name="calculator", args={"expression": "2+2"}, id="4"),
            ],
            [
                ToolCall(name="search", args={"query": "test3"}, id="5"),
                ToolCall(name="calculator", args={"expression": "3+3"}, id="6"),
            ],
            [],
        ]
    )

    # Limit search to 2 calls, but allow other tools to continue
    search_limiter = ToolCallLimitMiddleware(
        tool_name="search", thread_limit=2, allow_other_tools=True
    )

    agent = create_agent(
        model=model,
        tools=[search, calculator],
        middleware=[search_limiter],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Question")]},
        {"configurable": {"thread_id": "test_thread"}},
    )

    # Check that search was limited but calculator continued
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    # Count successful calculator calls (those that have "Result:" in content)
    calc_tool_messages = [
        m for m in tool_messages if "Result:" in m.content
    ]

    # Count search error messages (those that have "limit" in content)
    limit_error_messages = [
        m for m in tool_messages if "limit" in m.content
    ]

    # We should have 3 calculator calls that succeeded
    # and at least 1 search call that got a limit error
    assert len(calc_tool_messages) == 3  # All calculator calls should succeed
    assert len(limit_error_messages) >= 1  # At least one search hit the limit


def test_allow_other_tools_false():
    """Test that allow_other_tools=False ends execution immediately."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    @tool
    def calculator(expression: str) -> str:
        """Calculate an expression."""
        return f"Result: {expression}"

    # Create model that makes multiple tool calls
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "test1"}, id="1"),
                ToolCall(name="calculator", args={"expression": "1+1"}, id="2"),
            ],
            [
                ToolCall(name="search", args={"query": "test2"}, id="3"),
                ToolCall(name="calculator", args={"expression": "2+2"}, id="4"),
            ],
            [
                ToolCall(name="search", args={"query": "test3"}, id="5"),
            ],
            [],
        ]
    )

    # Limit search to 2 calls, end immediately when exceeded
    search_limiter = ToolCallLimitMiddleware(
        tool_name="search", thread_limit=2, allow_other_tools=False
    )

    agent = create_agent(
        model=model,
        tools=[search, calculator],
        middleware=[search_limiter],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Question")]},
        {"configurable": {"thread_id": "test_thread"}},
    )

    # Check that execution ended with a limit message
    last_message = result["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert "'search' tool" in last_message.content
    assert "Thread limit exceeded" in last_message.content
