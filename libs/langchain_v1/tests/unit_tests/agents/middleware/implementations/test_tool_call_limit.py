"""Unit tests for ToolCallLimitMiddleware."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.factory import create_agent
from langchain.agents.middleware.tool_call_limit import (
    ToolCallLimitExceededError,
    ToolCallLimitMiddleware,
    ToolCallLimitState,
)
from tests.unit_tests.agents.model import FakeToolCallingModel


def test_middleware_initialization_validation() -> None:
    """Test that middleware initialization validates parameters correctly."""
    # Test that at least one limit must be specified
    with pytest.raises(ValueError, match="At least one limit must be specified"):
        ToolCallLimitMiddleware()

    # Test valid initialization with both limits
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

    # Test exit behaviors
    for behavior in ["error", "end", "continue"]:
        middleware = ToolCallLimitMiddleware(thread_limit=5, exit_behavior=behavior)
        assert middleware.exit_behavior == behavior

    # Test invalid exit behavior
    with pytest.raises(ValueError, match="Invalid exit_behavior"):
        ToolCallLimitMiddleware(thread_limit=5, exit_behavior="invalid")  # type: ignore[arg-type]

    # Test run_limit exceeding thread_limit
    with pytest.raises(
        ValueError,
        match=r"run_limit .* cannot exceed thread_limit",
    ):
        ToolCallLimitMiddleware(thread_limit=3, run_limit=5)

    # Test run_limit equal to thread_limit (should be valid)
    middleware = ToolCallLimitMiddleware(thread_limit=5, run_limit=5)
    assert middleware.thread_limit == 5
    assert middleware.run_limit == 5

    # Test run_limit less than thread_limit (should be valid)
    middleware = ToolCallLimitMiddleware(thread_limit=5, run_limit=3)
    assert middleware.thread_limit == 5
    assert middleware.run_limit == 3


def test_middleware_name_property() -> None:
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


def test_middleware_unit_functionality() -> None:
    """Test that the middleware works as expected in isolation.

    Tests basic count tracking, thread limit, run limit, and limit-not-exceeded cases.
    """
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2, exit_behavior="end")
    runtime = None

    # Test when limits are not exceeded - counts should increment normally
    state = ToolCallLimitState(
        messages=[
            HumanMessage("Question"),
            AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ],
        thread_tool_call_count={},
        run_tool_call_count={},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["thread_tool_call_count"] == {"__all__": 1}
    assert result["run_tool_call_count"] == {"__all__": 1}
    assert "jump_to" not in result

    # Test thread limit exceeded (start at thread_limit so next call will exceed)
    state = ToolCallLimitState(
        messages=[
            HumanMessage("Question 2"),
            AIMessage("Response 2", tool_calls=[{"name": "search", "args": {}, "id": "3"}]),
        ],
        thread_tool_call_count={"__all__": 3},  # Already exceeds thread_limit=3
        run_tool_call_count={"__all__": 0},  # No calls yet
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    # Check the ToolMessage (sent to model - no thread/run details)
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.status == "error"
    assert "Tool call limit exceeded" in tool_msg.content
    # Should include "Do not" instruction
    assert "Do not" in tool_msg.content, (
        "Tool message should include 'Do not' instruction when limit exceeded"
    )
    # Check the final AI message (displayed to user - includes thread/run details)
    final_ai_msg = result["messages"][-1]
    assert isinstance(final_ai_msg, AIMessage)
    assert isinstance(final_ai_msg.content, str)
    assert "limit" in final_ai_msg.content.lower()
    assert "thread limit exceeded" in final_ai_msg.content.lower()
    # Thread count stays at 3 (blocked call not counted)
    assert result["thread_tool_call_count"] == {"__all__": 3}
    # Run count goes to 1 (includes blocked call)
    assert result["run_tool_call_count"] == {"__all__": 1}

    # Test run limit exceeded (thread count must be >= run count)
    state = ToolCallLimitState(
        messages=[
            HumanMessage("Question"),
            AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ],
        thread_tool_call_count={"__all__": 2},
        run_tool_call_count={"__all__": 2},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    # Check the final AI message includes run limit details
    final_ai_msg = result["messages"][-1]
    assert "run limit exceeded" in final_ai_msg.content
    assert "3/2 calls" in final_ai_msg.content
    # Check the tool message (sent to model) - should always include "Do not" instruction
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert "Tool call limit exceeded" in tool_msg.content
    assert "Do not" in tool_msg.content, (
        "Tool message should include 'Do not' instruction for both run and thread limits"
    )


def test_middleware_end_behavior_with_unrelated_parallel_tool_calls() -> None:
    """Test middleware 'end' behavior with unrelated parallel tool calls.

    Test that 'end' behavior raises NotImplementedError when there are parallel calls
    to unrelated tools.

    When limiting a specific tool with "end" behavior and the model proposes parallel calls
    to BOTH the limited tool AND other tools, we can't handle this scenario (we'd be stopping
    execution while other tools should run).
    """
    # Limit search tool specifically
    middleware = ToolCallLimitMiddleware(tool_name="search", thread_limit=1, exit_behavior="end")
    runtime = None

    # Test with search + calculator calls when search exceeds limit
    state = ToolCallLimitState(
        messages=[
            AIMessage(
                "Response",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "calculator", "args": {}, "id": "2"},
                ],
            ),
        ],
        thread_tool_call_count={"search": 1},
        run_tool_call_count={"search": 1},
    )

    with pytest.raises(
        NotImplementedError, match="Cannot end execution with other tool calls pending"
    ):
        middleware.after_model(state, runtime)  # type: ignore[arg-type]


def test_middleware_with_specific_tool() -> None:
    """Test middleware that limits a specific tool while ignoring others."""
    middleware = ToolCallLimitMiddleware(
        tool_name="search", thread_limit=2, run_limit=1, exit_behavior="end"
    )
    runtime = None

    # Test search tool exceeding run limit
    state = ToolCallLimitState(
        messages=[
            AIMessage("Response 2", tool_calls=[{"name": "search", "args": {}, "id": "3"}]),
        ],
        thread_tool_call_count={"search": 1},
        run_tool_call_count={"search": 1},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "search" in result["messages"][0].content.lower()

    # Test calculator tool - should be ignored by search-specific middleware
    state = ToolCallLimitState(
        messages=[
            AIMessage("Response", tool_calls=[{"name": "calculator", "args": {}, "id": "1"}] * 10),
        ],
        thread_tool_call_count={},
        run_tool_call_count={},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is None, "Calculator calls shouldn't be counted by search-specific middleware"


def test_middleware_error_behavior() -> None:
    """Test middleware error behavior.

    Test that middleware raises ToolCallLimitExceededError when configured with
    exit_behavior='error'.
    """
    middleware = ToolCallLimitMiddleware(thread_limit=2, exit_behavior="error")
    runtime = None

    state = ToolCallLimitState(
        messages=[AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}])],
        thread_tool_call_count={"__all__": 2},
        run_tool_call_count={"__all__": 2},
    )

    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.after_model(state, runtime)  # type: ignore[arg-type]

    error = exc_info.value
    # Thread count in error message shows hypothetical count (what it would have been)
    assert error.thread_count == 3
    assert error.thread_limit == 2
    # Run count includes the blocked call
    assert error.run_count == 3
    assert error.tool_name is None


def test_multiple_middleware_instances() -> None:
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
    ai_limit_messages = [
        msg
        for msg in result["messages"]
        if isinstance(msg, AIMessage)
        and isinstance(msg.content, str)
        and "limit" in msg.content.lower()
    ]
    assert len(ai_limit_messages) > 0, "Should have AI message explaining limit was exceeded"


def test_run_limit_with_multiple_human_messages() -> None:
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
    tool_messages = [msg for msg in result1["messages"] if isinstance(msg, ToolMessage)]
    successful_tool_msgs = [msg for msg in tool_messages if msg.status != "error"]
    error_tool_msgs = [msg for msg in tool_messages if msg.status == "error"]
    ai_limit_msgs = [
        msg
        for msg in result1["messages"]
        if isinstance(msg, AIMessage)
        and isinstance(msg.content, str)
        and "limit" in msg.content.lower()
        and not msg.tool_calls
    ]

    assert len(successful_tool_msgs) == 1, "Should have 1 successful tool execution (test1)"
    assert len(error_tool_msgs) == 1, "Should have 1 artificial error ToolMessage (test2)"
    assert len(ai_limit_msgs) == 1, "Should have AI limit message after test2 proposed"

    # Second invocation: run limit should reset, allowing continued execution
    result2 = agent.invoke(
        {"messages": [HumanMessage("Question 2")]},
        {"configurable": {"thread_id": "test_thread"}},
    )

    assert len(result2["messages"]) > len(result1["messages"]), (
        "Second invocation should add new messages, proving run limit reset"
    )


def test_exception_error_messages() -> None:
    """Test that error messages include expected information."""
    # Test for specific tool
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        raise ToolCallLimitExceededError(
            thread_count=5, run_count=3, thread_limit=4, run_limit=2, tool_name="search"
        )
    msg = str(exc_info.value)
    assert "search" in msg.lower()
    assert "5/4" in msg or "thread" in msg.lower()

    # Test for all tools
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        raise ToolCallLimitExceededError(
            thread_count=10, run_count=5, thread_limit=8, run_limit=None, tool_name=None
        )
    msg = str(exc_info.value)
    assert "10/8" in msg or "thread" in msg.lower()


def test_limit_reached_but_not_exceeded() -> None:
    """Test that limits are only triggered when exceeded (>), not when reached (==)."""
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2, exit_behavior="end")
    runtime = None

    # Test when limit is reached exactly (count = limit) - should not trigger
    state = ToolCallLimitState(
        messages=[AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}])],
        thread_tool_call_count={"__all__": 2},  # After +1 will be exactly 3
        run_tool_call_count={"__all__": 1},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert "jump_to" not in result
    assert result["thread_tool_call_count"]["__all__"] == 3

    # Test when limit is exceeded (count > limit) - should trigger
    state = ToolCallLimitState(
        messages=[AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "1"}])],
        thread_tool_call_count={"__all__": 3},  # After +1 will be 4 > 3
        run_tool_call_count={"__all__": 1},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert "jump_to" in result
    assert result["jump_to"] == "end"


def test_exit_behavior_continue() -> None:
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

    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]

    # Verify search has 2 successful + 1 blocked, calculator has all 3 successful
    successful_search_msgs = [msg for msg in tool_messages if "Search:" in msg.content]
    blocked_search_msgs = [
        msg
        for msg in tool_messages
        if isinstance(msg.content, str)
        and "limit" in msg.content.lower()
        and "search" in msg.content.lower()
    ]
    successful_calc_msgs = [msg for msg in tool_messages if "Calc:" in msg.content]

    assert len(successful_search_msgs) == 2, "Should have 2 successful search calls"
    assert len(blocked_search_msgs) == 1, "Should have 1 blocked search call with limit error"
    assert len(successful_calc_msgs) == 3, "All calculator calls should succeed"


def test_thread_count_excludes_blocked_run_calls() -> None:
    """Test that thread count only includes allowed calls, not blocked run-scoped calls.

    When run_limit is lower than thread_limit and multiple parallel calls are made,
    only the allowed calls should increment the thread count.

    Example: If run_limit=1 and 3 parallel calls are made, thread count should be 1
    (not 3) because the other 2 were blocked by the run limit.
    """
    # Set run_limit=1, thread_limit=10 (much higher)
    middleware = ToolCallLimitMiddleware(thread_limit=10, run_limit=1, exit_behavior="continue")
    runtime = None

    # Make 3 parallel tool calls - only 1 should be allowed by run_limit
    state = ToolCallLimitState(
        messages=[
            AIMessage(
                "Response",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "search", "args": {}, "id": "2"},
                    {"name": "search", "args": {}, "id": "3"},
                ],
            )
        ],
        thread_tool_call_count={},
        run_tool_call_count={},
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None

    # Thread count should be 1 (only the allowed call)
    assert result["thread_tool_call_count"]["__all__"] == 1, (
        "Thread count should only include the 1 allowed call, not the 2 blocked calls"
    )
    # Run count should be 3 (all attempted calls)
    assert result["run_tool_call_count"]["__all__"] == 3, (
        "Run count should include all 3 attempted calls"
    )

    # Verify 2 error messages were created for blocked calls
    assert "messages" in result
    error_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(error_messages) == 2, "Should have 2 error messages for the 2 blocked calls"


def test_unified_error_messages() -> None:
    """Test that error messages instruct model not to call again for both run and thread limits.

    Previously, only thread limit messages included 'Do not' instruction.
    Now both run and thread limit messages should include it.
    """
    middleware = ToolCallLimitMiddleware(thread_limit=10, run_limit=1, exit_behavior="continue")
    runtime = None

    # Test with run limit exceeded (thread limit not exceeded)
    state = ToolCallLimitState(
        messages=[AIMessage("Response", tool_calls=[{"name": "search", "args": {}, "id": "2"}])],
        thread_tool_call_count={"__all__": 1},  # Under thread limit
        run_tool_call_count={"__all__": 1},  # At run limit, next call will exceed
    )
    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None

    # Check the error message includes "Do not" instruction
    error_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(error_messages) == 1
    error_content = error_messages[0].content
    assert "Do not" in error_content, (
        "Run limit error message should include 'Do not' instruction to guide model behavior"
    )


def test_end_behavior_creates_artificial_messages() -> None:
    """Test that 'end' behavior creates an AI message explaining why execution stopped.

    Verifies that when limit is exceeded with exit_behavior='end', the middleware:
    1. Injects an artificial error ToolMessage for the blocked tool call
    2. Adds an AI message explaining the limit to the user
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

    # Verify AI message explaining the limit (displayed to user - includes thread/run details)
    ai_limit_messages = [
        msg
        for msg in result["messages"]
        if isinstance(msg, AIMessage)
        and isinstance(msg.content, str)
        and "limit" in msg.content.lower()
        and not msg.tool_calls
    ]
    assert len(ai_limit_messages) == 1, "Should have exactly one AI message explaining the limit"

    assert isinstance(ai_limit_messages[0].content, str)
    ai_msg_content = ai_limit_messages[0].content.lower()
    assert "thread limit exceeded" in ai_msg_content or "run limit exceeded" in ai_msg_content, (
        "AI message should include thread/run limit details for the user"
    )

    # Verify tool message counts
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    successful_tool_msgs = [msg for msg in tool_messages if msg.status != "error"]
    error_tool_msgs = [msg for msg in tool_messages if msg.status == "error"]

    assert len(successful_tool_msgs) == 2, "Should have 2 successful tool messages (q1, q2)"
    assert len(error_tool_msgs) == 1, "Should have 1 artificial error tool message (q3)"

    # Verify the error tool message (sent to model - no thread/run details, includes instruction)
    error_msg_content = error_tool_msgs[0].content
    assert "Tool call limit exceeded" in error_msg_content
    assert "Do not" in error_msg_content, (
        "Tool message should instruct model not to call tool again"
    )


def test_parallel_tool_calls_with_limit_continue_mode() -> None:
    """Test parallel tool calls with a limit of 1 in 'continue' mode.

    When the model proposes 3 tool calls with a limit of 1:
    - The first call should execute successfully
    - The 2nd and 3rd calls should be blocked with error ToolMessages
    - Execution should continue (no jump_to)
    """

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results: {query}"

    # Model proposes 3 parallel search calls in a single AIMessage
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "q1"}, id="1"),
                ToolCall(name="search", args={"query": "q2"}, id="2"),
                ToolCall(name="search", args={"query": "q3"}, id="3"),
            ],
            [],  # Model stops after seeing the errors
        ]
    )

    limiter = ToolCallLimitMiddleware(thread_limit=1, exit_behavior="continue")
    agent = create_agent(
        model=model, tools=[search], middleware=[limiter], checkpointer=InMemorySaver()
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}}
    )
    messages = result["messages"]

    # Verify tool message counts
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    successful_tool_messages = [msg for msg in tool_messages if msg.status != "error"]
    error_tool_messages = [msg for msg in tool_messages if msg.status == "error"]

    assert len(successful_tool_messages) == 1, "Should have 1 successful tool message (q1)"
    assert len(error_tool_messages) == 2, "Should have 2 blocked tool messages (q2, q3)"

    # Verify the successful call is q1
    assert "q1" in successful_tool_messages[0].content

    # Verify error messages explain the limit
    for error_msg in error_tool_messages:
        assert isinstance(error_msg.content, str)
        assert "limit" in error_msg.content.lower()

    # Verify execution continued (no early termination)
    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
    # Should have: initial AI message with 3 tool calls, then final AI message (no tool calls)
    assert len(ai_messages) >= 2


def test_parallel_tool_calls_with_limit_end_mode() -> None:
    """Test parallel tool calls with a limit of 1 in 'end' mode.

    When the model proposes 3 tool calls with a limit of 1:
    - The first call would be allowed (within limit)
    - The 2nd and 3rd calls exceed the limit and get blocked with error ToolMessages
    - Execution stops immediately (jump_to: end) so NO tools actually execute
    - An AI message explains why execution stopped
    """

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results: {query}"

    # Model proposes 3 parallel search calls
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "q1"}, id="1"),
                ToolCall(name="search", args={"query": "q2"}, id="2"),
                ToolCall(name="search", args={"query": "q3"}, id="3"),
            ],
            [],
        ]
    )

    limiter = ToolCallLimitMiddleware(thread_limit=1, exit_behavior="end")
    agent = create_agent(
        model=model, tools=[search], middleware=[limiter], checkpointer=InMemorySaver()
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}}
    )
    messages = result["messages"]

    # Verify tool message counts
    # With "end" behavior, when we jump to end, NO tools execute (not even allowed ones)
    # We only get error ToolMessages for the 2 blocked calls
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    successful_tool_messages = [msg for msg in tool_messages if msg.status != "error"]
    error_tool_messages = [msg for msg in tool_messages if msg.status == "error"]

    assert len(successful_tool_messages) == 0, "No tools execute when we jump to end"
    assert len(error_tool_messages) == 2, "Should have 2 blocked tool messages (q2, q3)"

    # Verify error tool messages (sent to model - include "Do not" instruction)
    for error_msg in error_tool_messages:
        assert "Tool call limit exceeded" in error_msg.content
        assert "Do not" in error_msg.content

    # Verify AI message explaining why execution stopped
    # (displayed to user - includes thread/run details)
    ai_limit_messages = [
        msg
        for msg in messages
        if isinstance(msg, AIMessage)
        and isinstance(msg.content, str)
        and "limit" in msg.content.lower()
        and not msg.tool_calls
    ]
    assert len(ai_limit_messages) == 1, "Should have exactly one AI message explaining the limit"

    assert isinstance(ai_limit_messages[0].content, str)
    ai_msg_content = ai_limit_messages[0].content.lower()
    assert "thread limit exceeded" in ai_msg_content or "run limit exceeded" in ai_msg_content, (
        "AI message should include thread/run limit details for the user"
    )


def test_parallel_mixed_tool_calls_with_specific_tool_limit() -> None:
    """Test parallel calls to different tools when limiting a specific tool.

    When limiting 'search' to 1 call, and model proposes 3 search + 2 calculator calls:
    - First search call should execute
    - Other 2 search calls should be blocked
    - All calculator calls should execute (not limited)
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
                ToolCall(name="search", args={"query": "q2"}, id="3"),
                ToolCall(name="calculator", args={"expression": "2+2"}, id="4"),
                ToolCall(name="search", args={"query": "q3"}, id="5"),
            ],
            [],
        ]
    )

    search_limiter = ToolCallLimitMiddleware(
        tool_name="search", thread_limit=1, exit_behavior="continue"
    )
    agent = create_agent(
        model=model,
        tools=[search, calculator],
        middleware=[search_limiter],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}}
    )
    messages = result["messages"]

    search_success = []
    search_blocked = []
    calc_success = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        assert isinstance(m.content, str)
        if "Search:" in m.content:
            search_success.append(m)
        if "limit" in m.content.lower() and "search" in m.content.lower():
            search_blocked.append(m)
        if "Calc:" in m.content:
            calc_success.append(m)

    assert len(search_success) == 1, "Should have 1 successful search call"
    assert len(search_blocked) == 2, "Should have 2 blocked search calls"
    assert len(calc_success) == 2, "All calculator calls should succeed (not limited)"
