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
    """Test that the middleware works as expected in isolation.

    Tests basic count tracking, thread limit, run limit, and limit-not-exceeded cases.
    """
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2, exit_behavior="end")
    runtime = None

    # Test when limits are not exceeded - counts should increment normally
    state = {
        "messages": [
            HumanMessage("Question"),
            AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ],
        "thread_tool_call_count": {},
        "run_tool_call_count": {},
    }
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["thread_tool_call_count"] == {"__all__": 1}
    assert result["run_tool_call_count"] == {"__all__": 1}
    assert "jump_to" not in result

    # Test thread limit exceeded
    state = {
        "messages": [
            HumanMessage("Question 2"),
            AIMessage("Response 2", tool_calls=[{"name": "search", "args": {}, "id": "3"}]),
        ],
        "thread_tool_call_count": {"__all__": 3},
        "run_tool_call_count": {"__all__": 1},
    }
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    # Check the ToolMessage (sent to model - no thread/run details)
    tool_msg = result["messages"][0]
    assert tool_msg.status == "error"
    assert "Tool call limit exceeded" in tool_msg.content
    # When thread limit exceeded, should include "Do not" instruction
    assert "Do not" in tool_msg.content, (
        "Tool message should include 'Do not' instruction when thread limit exceeded"
    )
    # Check the final AI message (displayed to user - includes thread/run details)
    assert "limit" in result["messages"][-1].content.lower()
    assert "thread limit exceeded" in result["messages"][-1].content.lower()
    assert result["thread_tool_call_count"] == {"__all__": 4}

    # Test run limit exceeded
    state = {
        "messages": [
            HumanMessage("Question"),
            AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ],
        "thread_tool_call_count": {"__all__": 0},
        "run_tool_call_count": {"__all__": 2},
    }
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    # Check the final AI message includes run limit details
    assert "run limit exceeded" in result["messages"][-1].content
    assert "3/2 calls" in result["messages"][-1].content
    # Check the tool message (sent to model) - when only run limit exceeded,
    # should not include "Do not" instruction since run is ending anyway
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert "Tool call limit exceeded" in tool_msg.content
    assert "Do not" not in tool_msg.content, (
        "Tool message should not include 'Do not' instruction when only run limit exceeded"
    )


def test_middleware_end_behavior_with_unrelated_parallel_tool_calls():
    """Test that 'end' behavior raises NotImplementedError when there are parallel calls to unrelated tools.

    When limiting a specific tool with "end" behavior and the model proposes parallel calls
    to BOTH the limited tool AND other tools, we can't handle this scenario (we'd be stopping
    execution while other tools should run).
    """
    # Limit search tool specifically
    middleware = ToolCallLimitMiddleware(tool_name="search", thread_limit=1, exit_behavior="end")
    runtime = None

    # Test with search + calculator calls when search exceeds limit
    state = {
        "messages": [
            AIMessage(
                "Response",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "calculator", "args": {}, "id": "2"},
                ],
            ),
        ],
        "thread_tool_call_count": {"search": 1},
        "run_tool_call_count": {"search": 1},
    }

    with pytest.raises(
        NotImplementedError, match="Cannot end execution with other tool calls pending"
    ):
        middleware.after_model(state, runtime)  # type: ignore[arg-type]


def test_middleware_with_specific_tool():
    """Test middleware that limits a specific tool while ignoring others."""
    middleware = ToolCallLimitMiddleware(
        tool_name="search", thread_limit=2, run_limit=1, exit_behavior="end"
    )
    runtime = None

    # Test search tool exceeding run limit
    state = {
        "messages": [
            AIMessage("Response 2", tool_calls=[{"name": "search", "args": {}, "id": "3"}]),
        ],
        "thread_tool_call_count": {"search": 1},
        "run_tool_call_count": {"search": 1},
    }
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "search" in result["messages"][0].content.lower()

    # Test calculator tool - should be ignored by search-specific middleware
    state = {
        "messages": [
            AIMessage("Response", tool_calls=[{"name": "calculator", "args": {}, "id": "1"}] * 10),
        ],
        "thread_tool_call_count": {},
        "run_tool_call_count": {},
    }
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is None, "Calculator calls shouldn't be counted by search-specific middleware"


def test_middleware_error_behavior():
    """Test that middleware raises ToolCallLimitExceededError when configured with exit_behavior='error'."""
    middleware = ToolCallLimitMiddleware(thread_limit=2, exit_behavior="error")
    runtime = None

    state = {
        "messages": [AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}])],
        "thread_tool_call_count": {"__all__": 2},
        "run_tool_call_count": {"__all__": 2},
    }

    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.after_model(state, runtime)  # type: ignore[arg-type]

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
    """Test that run limits reset between invocations.

    Verifies that when using run_limit, the count resets for each new user message,
    allowing execution to continue across multiple invocations in the same thread.
    """

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

    middleware = ToolCallLimitMiddleware(run_limit=1, exit_behavior="end")
    agent = create_agent(
        model=model, tools=[search], middleware=[middleware], checkpointer=InMemorySaver()
    )

    # First invocation: test1 executes successfully, test2 exceeds limit
    result1 = agent.invoke(
        {"messages": [HumanMessage("Question 1")]},
        {"configurable": {"thread_id": "test_thread"}},
    )
    messages = result1["messages"]
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

    # Second invocation: run limit should reset, allowing continued execution
    result2 = agent.invoke(
        {"messages": [HumanMessage("Question 2")]},
        {"configurable": {"thread_id": "test_thread"}},
    )

    assert len(result2["messages"]) > len(result1["messages"]), (
        "Second invocation should add new messages, proving run limit reset"
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
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2, exit_behavior="end")
    runtime = None

    # Test when limit is reached exactly (count = limit) - should not trigger
    state = {
        "messages": [AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}])],
        "thread_tool_call_count": {"__all__": 2},  # After +1 will be exactly 3
        "run_tool_call_count": {"__all__": 1},
    }
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert "jump_to" not in result
    assert result["thread_tool_call_count"]["__all__"] == 3

    # Test when limit is exceeded (count > limit) - should trigger
    state = {
        "messages": [AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}])],
        "thread_tool_call_count": {"__all__": 3},  # After +1 will be 4 > 3
        "run_tool_call_count": {"__all__": 1},
    }
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert "jump_to" in result
    assert result["jump_to"] == "end"


def test_exit_behavior_continue():
    """Test that exit_behavior='continue' blocks only the exceeded tool, not others.

    Verifies that when a specific tool hits its limit, it gets blocked with error messages
    while other tools continue to execute normally.
    """

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Search: {query}"

    @tool
    def calculator(expression: str) -> str:
        """Calculate an expression."""
        return f"Calc: {expression}"

    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "q1"}, id="1"),
                ToolCall(name="calculator", args={"expression": "1+1"}, id="2"),
            ],
            [
                ToolCall(name="search", args={"query": "q2"}, id="3"),
                ToolCall(name="calculator", args={"expression": "2+2"}, id="4"),
            ],
            [
                ToolCall(name="search", args={"query": "q3"}, id="5"),  # Should be blocked
                ToolCall(name="calculator", args={"expression": "3+3"}, id="6"),  # Should work
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

    messages = result["messages"]
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]

    # Verify search has 2 successful + 1 blocked, calculator has all 3 successful
    search_success = [m for m in tool_messages if "Search:" in m.content]
    search_blocked = [
        m for m in tool_messages if "limit" in m.content.lower() and "search" in m.content.lower()
    ]
    calc_success = [m for m in tool_messages if "Calc:" in m.content]

    assert len(search_success) == 2, "Should have 2 successful search calls"
    assert len(search_blocked) == 1, "Should have 1 blocked search call with limit error"
    assert len(calc_success) == 3, "All calculator calls should succeed"


def test_end_behavior_creates_artificial_messages():
    """Test that 'end' behavior creates an AI message explaining why execution stopped.

    Verifies that when limit is exceeded with exit_behavior='end', the middleware:
    1. Injects an artificial error ToolMessage for the blocked tool call
    2. Adds an AI message instructing the model not to call the tool again
    3. Jumps to end, stopping execution
    """

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results: {query}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "q1"}, id="1")],
            [ToolCall(name="search", args={"query": "q2"}, id="2")],
            [ToolCall(name="search", args={"query": "q3"}, id="3")],  # Exceeds limit
            [],
        ]
    )

    limiter = ToolCallLimitMiddleware(thread_limit=2, exit_behavior="end")
    agent = create_agent(
        model=model, tools=[search], middleware=[limiter], checkpointer=InMemorySaver()
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}}
    )
    messages = result["messages"]

    # Verify AI message explaining the limit (displayed to user - includes thread/run details)
    ai_limit_messages = [
        msg
        for msg in messages
        if isinstance(msg, AIMessage) and "limit" in msg.content.lower() and not msg.tool_calls
    ]
    assert len(ai_limit_messages) == 1, "Should have exactly one AI message explaining the limit"

    ai_msg_content = ai_limit_messages[0].content.lower()
    assert "thread limit exceeded" in ai_msg_content or "run limit exceeded" in ai_msg_content, (
        "AI message should include thread/run limit details for the user"
    )

    # Verify tool message counts
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    successful_tool_messages = [msg for msg in tool_messages if msg.status != "error"]
    error_tool_messages = [msg for msg in tool_messages if msg.status == "error"]

    assert len(successful_tool_messages) == 2, "Should have 2 successful tool messages (q1, q2)"
    assert len(error_tool_messages) == 1, "Should have 1 artificial error tool message (q3)"

    # Verify the error tool message (sent to model - no thread/run details, includes instruction)
    error_msg_content = error_tool_messages[0].content
    assert "Tool call limit exceeded" in error_msg_content
    assert "Do not" in error_msg_content, (
        "Tool message should instruct model not to call tool again"
    )
