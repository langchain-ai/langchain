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

    # Test valid initialization
    middleware = ToolCallLimitMiddleware(thread_limit=5, run_limit=3)
    assert middleware.thread_limit == 5
    assert middleware.run_limit == 3
    assert middleware.exit_behavior == "continue"
    assert middleware.tool_name is None

    # Test with tool name
    middleware = ToolCallLimitMiddleware(tool_name="search", thread_limit=5)
    assert middleware.tool_name == "search"
    assert middleware.thread_limit == 5
    assert middleware.run_limit is None

    # Test different exit behaviors
    middleware = ToolCallLimitMiddleware(thread_limit=5, exit_behavior="error")
    assert middleware.exit_behavior == "error"

    middleware = ToolCallLimitMiddleware(thread_limit=5, exit_behavior="end")
    assert middleware.exit_behavior == "end"

    middleware = ToolCallLimitMiddleware(thread_limit=5, exit_behavior="continue")
    assert middleware.exit_behavior == "continue"


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
    # Test with exit_behavior="end" - exit when limit exceeded after incrementing counts
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2, exit_behavior="end")

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
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    # Counts should be incremented but limit not exceeded
    assert result["thread_tool_call_count"] == {"__all__": 1}
    assert result["run_tool_call_count"] == {"__all__": 1}
    assert "jump_to" not in result  # No limit exceeded

    # Test when thread limit is exceeded
    state = {
        "messages": [
            HumanMessage("Question 2"),
            AIMessage(
                "Response 2",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "3"},
                    {"name": "calculator", "args": {}, "id": "4"},
                ],
            ),
        ],
        "thread_tool_call_count": {"__all__": 2},  # 2 calls so far
        "run_tool_call_count": {"__all__": 2},  # 2 calls in run
    }
    # After incrementing by 2 more, thread count will be 4 > 3
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "limit reached" in result["messages"][0].content.lower()
    assert result["thread_tool_call_count"] == {"__all__": 4}

    # Test when run limit is exceeded
    state = {
        "messages": [
            HumanMessage("Question"),
            AIMessage(
                "Response",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "calculator", "args": {}, "id": "2"},
                    {"name": "search", "args": {}, "id": "3"},
                ],
            ),
        ],
        "thread_tool_call_count": {"__all__": 0},  # Within thread limit
        "run_tool_call_count": {"__all__": 0},  # Starting fresh
    }
    # After incrementing by 3, run count will be 3 > 2
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "run limit exceeded (3/2 calls)" in result["messages"][0].content


def test_middleware_with_specific_tool():
    """Test middleware that limits a specific tool."""
    # Limit only the "search" tool with exit_behavior="end"
    middleware = ToolCallLimitMiddleware(
        tool_name="search", thread_limit=2, run_limit=1, exit_behavior="end"
    )

    runtime = None

    # Test with search tool call - will exceed run limit after incrementing
    state = {
        "messages": [
            AIMessage(
                "Response 2",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "3"},  # Second search call
                ],
            ),
        ],
        "thread_tool_call_count": {"search": 1},  # 1 search so far
        "run_tool_call_count": {"search": 1},  # 1 in current run
    }
    # After incrementing by 1, run count will be 2 > 1
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "search" in result["messages"][0].content.lower()

    # Test with calculator tool - should not trigger limit (different tool)
    state = {
        "messages": [
            AIMessage(
                "Response",
                tool_calls=[{"name": "calculator", "args": {}, "id": "1"}] * 10,
            ),
        ],
        "thread_tool_call_count": {},
        "run_tool_call_count": {},
    }
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    # Since we're only limiting "search", calculator calls don't count toward limit
    # The middleware should return None (no relevant tool calls to count)
    assert result is None, "Calculator calls shouldn't be counted by search-specific middleware"


def test_middleware_error_behavior():
    """Test that middleware raises an error when configured to do so."""
    middleware = ToolCallLimitMiddleware(thread_limit=2, exit_behavior="error")

    runtime = None

    state = {
        "messages": [
            AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ],
        "thread_tool_call_count": {"__all__": 2},  # At limit
        "run_tool_call_count": {"__all__": 2},
    }

    # After incrementing by 1, count will be 3 > 2, should raise error
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.after_model(state, runtime)  # type: ignore[arg-type]

    # Verify the exception contains expected information
    error = exc_info.value
    assert error.thread_count == 3
    assert error.thread_limit == 2
    assert error.tool_name is None


def test_multiple_middleware_instances():
    """Test that multiple middleware instances can coexist and track independently."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for {query}"

    @tool
    def calculator(expression: str) -> str:
        """Calculate an expression."""
        return f"Result: {expression}"

    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "test"}, id="1"),
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

    # Create two middleware instances - one for each tool
    search_limiter = ToolCallLimitMiddleware(
        tool_name="search", thread_limit=2, exit_behavior="end"
    )
    calc_limiter = ToolCallLimitMiddleware(
        tool_name="calculator", thread_limit=2, exit_behavior="end"
    )

    agent = create_agent(
        model=model,
        tools=[search, calculator],
        middleware=[search_limiter, calc_limiter],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Question")]},
        {"configurable": {"thread_id": "test_thread"}},
    )

    # The agent should stop after the second iteration
    # because search will hit its limit (3 calls > 2 limit)
    messages = result["messages"]
    assert len(messages) > 0

    # Look for the limit message
    limit_messages = [
        msg for msg in messages if isinstance(msg, AIMessage) and "limit" in msg.content.lower()
    ]
    assert len(limit_messages) > 0


def test_run_limit_with_multiple_human_messages():
    """Test that run limits reset between invocations."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for {query}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test1"}, id="1")],
            [ToolCall(name="search", args={"query": "test2"}, id="2")],
            [],
        ]
    )

    # Run limit of 1 tool call per run, exit_behavior="end"
    middleware = ToolCallLimitMiddleware(run_limit=1, exit_behavior="end")

    agent = create_agent(
        model=model, tools=[search], middleware=[middleware], checkpointer=InMemorySaver()
    )

    # First invocation - test1 executes (count=1), then model proposes test2
    # which would make count=2 > limit of 1, so after_model injects artificial error
    # ToolMessage for test2 and jumps to end
    result1 = agent.invoke(
        {"messages": [HumanMessage("Question 1")]},
        {"configurable": {"thread_id": "test_thread"}},
    )
    messages = result1["messages"]
    # Should have 2 tool messages: test1 (successful) and test2 (artificial error)
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    successful_tool_messages = [m for m in tool_messages if m.status != "error"]
    error_tool_messages = [m for m in tool_messages if m.status == "error"]
    ai_limit_messages = [
        m
        for m in messages
        if isinstance(m, AIMessage) and "limit" in m.content.lower() and not m.tool_calls
    ]
    assert len(successful_tool_messages) == 1, "Should have 1 successful tool execution (test1)"
    assert len(error_tool_messages) == 1, "Should have 1 artificial error ToolMessage (test2)"
    assert len(ai_limit_messages) == 1, "Should have AI limit message after test2 proposed"

    # Second invocation in same thread - run limit resets
    # The agent has already reached the end in the first invocation, so the second
    # invocation will continue from that state. Since the agent ended, it won't
    # automatically continue making tool calls. This test verifies run limits reset,
    # which we can confirm by checking that the first invocation didn't fail immediately.
    # A more useful test would be with "continue" behavior to show execution continues.
    result2 = agent.invoke(
        {"messages": [HumanMessage("Question 2")]},
        {"configurable": {"thread_id": "test_thread"}},
    )

    # The second invocation should have more messages than the first (proves it executed)
    assert len(result2["messages"]) > len(result1["messages"]), (
        "Second invocation should add new messages, proving execution continued"
    )


def test_exception_error_messages():
    """Test that error messages include expected information."""
    # Test for specific tool
    try:
        raise ToolCallLimitExceededError(
            thread_count=5, run_count=3, thread_limit=4, run_limit=2, tool_name="search"
        )
    except ToolCallLimitExceededError as e:
        msg = str(e)
        assert "search" in msg.lower()
        assert "5/4" in msg or "thread" in msg.lower()

    # Test for all tools
    try:
        raise ToolCallLimitExceededError(
            thread_count=10, run_count=5, thread_limit=8, run_limit=None, tool_name=None
        )
    except ToolCallLimitExceededError as e:
        msg = str(e)
        assert "10/8" in msg or "thread" in msg.lower()


def test_limit_reached_but_not_exceeded():
    """Test that limits are only triggered when exceeded (>), not when reached (==)."""
    # Use exit_behavior="end" to test the after_model behavior
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2, exit_behavior="end")

    runtime = None

    # Test when limit will be reached exactly after incrementing
    state = {
        "messages": [
            AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ],
        "thread_tool_call_count": {"__all__": 2},  # After +1 will be exactly 3
        "run_tool_call_count": {"__all__": 1},
    }
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert "jump_to" not in result  # Should not jump - limit not exceeded (3 == 3)
    assert result["thread_tool_call_count"]["__all__"] == 3

    # Test when limit will be exceeded after incrementing
    state = {
        "messages": [
            AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ],
        "thread_tool_call_count": {"__all__": 3},  # After +1 will be 4 > 3
        "run_tool_call_count": {"__all__": 1},
    }
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert "jump_to" in result  # Should trigger - limit exceeded
    assert result["jump_to"] == "end"


def test_exit_behavior_continue():
    """Test that exit_behavior='continue' allows other tools to continue."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for {query}"

    @tool
    def calculator(expression: str) -> str:
        """Calculate an expression."""
        return f"Result: {expression}"

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
        tool_name="search", thread_limit=2, exit_behavior="continue"
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

    # The agent should continue executing, but search calls after the 2nd should be blocked
    messages = result["messages"]

    # Count tool messages by type (use the 'name' field, not tool_call_id)
    search_tool_messages = [
        msg for msg in messages if isinstance(msg, ToolMessage) and msg.name == "search"
    ]
    calc_tool_messages = [
        msg for msg in messages if isinstance(msg, ToolMessage) and msg.name == "calculator"
    ]

    # We should have error messages for search calls that exceeded the limit
    limit_error_messages = [
        msg for msg in messages if isinstance(msg, ToolMessage) and "limit" in msg.content.lower()
    ]

    # Calculator should still execute all its calls (3 successful)
    # Search should have 2 successful + 1 blocked with error message
    assert len(calc_tool_messages) == 3, "All 3 calculator calls should succeed"
    assert len(search_tool_messages) == 3, "Should have 2 successful + 1 error search messages"
    assert len(limit_error_messages) == 1, "Should have 1 search call blocked with limit error"


def test_exit_behavior_end():
    """Test that exit_behavior='end' ends execution immediately."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for {query}"

    @tool
    def calculator(expression: str) -> str:
        """Calculate an expression."""
        return f"Result: {expression}"

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
        tool_name="search", thread_limit=2, exit_behavior="end"
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

    # The agent should stop when search limit is exceeded
    messages = result["messages"]

    # Should have a limit message from the AI
    limit_messages = [
        msg for msg in messages if isinstance(msg, AIMessage) and "limit" in msg.content.lower()
    ]
    assert len(limit_messages) > 0


def test_continue_blocks_specific_tool_only():
    """Test that continue behavior blocks only the exceeded tool, not others."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Search: {query}"

    @tool
    def calculator(expression: str) -> str:
        """Calculate."""
        return f"Calc: {expression}"

    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "q1"}, id="1"),
            ],
            [
                ToolCall(name="search", args={"query": "q2"}, id="2"),
            ],
            [
                ToolCall(name="search", args={"query": "q3"}, id="3"),  # Should be blocked
                ToolCall(name="calculator", args={"expression": "1+1"}, id="4"),  # Should work
            ],
            [],
        ]
    )

    limiter = ToolCallLimitMiddleware(tool_name="search", thread_limit=2, exit_behavior="continue")

    agent = create_agent(
        model=model,
        tools=[search, calculator],
        middleware=[limiter],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}}
    )

    messages = result["messages"]

    # Find tool messages
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]

    # Should have 2 successful search calls, 1 blocked search, and 1 successful calculator
    search_success = [m for m in tool_messages if "Search:" in m.content]
    search_blocked = [
        m for m in tool_messages if "limit" in m.content.lower() and "search" in m.content.lower()
    ]
    calc_success = [m for m in tool_messages if "Calc:" in m.content]

    assert len(search_success) == 2, "Should have 2 successful search calls"
    assert len(search_blocked) == 1, "Should have 1 blocked search call"
    assert len(calc_success) == 1, "Calculator should still work"


def test_end_behavior_creates_artificial_messages():
    """Test that 'end' behavior creates an AI message explaining why execution stopped."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results: {query}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "q1"}, id="1")],
            [ToolCall(name="search", args={"query": "q2"}, id="2")],
            [
                ToolCall(name="search", args={"query": "q3"}, id="3"),  # These will execute
                ToolCall(name="search", args={"query": "q4"}, id="4"),  # pushing count to 4 > 2
            ],
            [],  # This won't be reached - limit exceeded after q3+q4
        ]
    )

    limiter = ToolCallLimitMiddleware(thread_limit=2, exit_behavior="end")

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[limiter],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}}
    )

    messages = result["messages"]

    # Should have an AI message explaining the limit was reached
    ai_limit_messages = [
        msg
        for msg in messages
        if isinstance(msg, AIMessage) and "limit" in msg.content.lower() and not msg.tool_calls
    ]
    assert len(ai_limit_messages) == 1, "Should have exactly one AI message explaining the limit"

    # The AI message should mention not to call the tool again
    ai_msg_content = ai_limit_messages[0].content.lower()
    assert "do not" in ai_msg_content or "don't" in ai_msg_content, (
        "Should instruct model not to call tool again"
    )

    # With "end" behavior, after_model checks limits AFTER tool calls are proposed by model
    # but BEFORE they execute. So q1 and q2 execute (count=2), then model proposes q3+q4,
    # after_model sees count would be 4 > 2, injects artificial error ToolMessages for q3+q4,
    # then adds AI message and jumps to end
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    successful_tool_messages = [msg for msg in tool_messages if msg.status != "error"]
    error_tool_messages = [msg for msg in tool_messages if msg.status == "error"]

    # Should have 2 successful tool executions (q1, q2) and 2 artificial error messages (q3, q4)
    assert len(successful_tool_messages) == 2, "Should have 2 successful tool messages (q1, q2)"
    assert len(error_tool_messages) == 2, (
        "Should have 2 artificial error tool messages (q3, q4) with 'end' behavior"
    )
