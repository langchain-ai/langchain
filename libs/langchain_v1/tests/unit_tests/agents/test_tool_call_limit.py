"""Unit tests for ToolCallLimitMiddleware."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.factory import create_agent
from langchain.agents.middleware.tool_call_limit import (
    ToolCallLimitExceededError,
    ToolCallLimitMiddleware,
    _count_tool_calls_in_messages,
    _get_run_messages,
)
from tests.unit_tests.agents.test_middleware_agent import FakeToolCallingModel


def test_count_tool_calls_in_messages():
    """Test counting tool calls in messages."""
    # Test with no messages
    assert _count_tool_calls_in_messages([]) == 0

    # Test with messages but no tool calls
    messages = [
        HumanMessage("Hello"),
        AIMessage("Hi there"),
    ]
    assert _count_tool_calls_in_messages(messages) == 0

    # Test with tool calls
    messages = [
        HumanMessage("Search for something"),
        AIMessage(
            "Searching...",
            tool_calls=[
                {"name": "search", "args": {"query": "test"}, "id": "1"},
                {"name": "calculator", "args": {"expression": "1+1"}, "id": "2"},
            ],
        ),
        ToolMessage("Result 1", tool_call_id="1"),
        ToolMessage("Result 2", tool_call_id="2"),
        AIMessage(
            "More searching...",
            tool_calls=[
                {"name": "search", "args": {"query": "another"}, "id": "3"},
            ],
        ),
    ]
    # Total: 3 tool calls (2 from first AI message, 1 from second)
    assert _count_tool_calls_in_messages(messages) == 3

    # Test filtering by tool name
    assert _count_tool_calls_in_messages(messages, tool_name="search") == 2
    assert _count_tool_calls_in_messages(messages, tool_name="calculator") == 1
    assert _count_tool_calls_in_messages(messages, tool_name="nonexistent") == 0


def test_get_run_messages():
    """Test extracting run messages after last HumanMessage."""
    # Test with no messages
    assert _get_run_messages([]) == []

    # Test with no HumanMessage
    messages = [
        AIMessage("Hello"),
        AIMessage("World"),
    ]
    assert _get_run_messages(messages) == messages

    # Test with HumanMessage at the end
    messages = [
        AIMessage("Previous"),
        HumanMessage("New question"),
    ]
    assert _get_run_messages(messages) == []

    # Test with messages after HumanMessage
    messages = [
        AIMessage("Previous"),
        ToolMessage("Previous result", tool_call_id="0"),
        HumanMessage("New question"),
        AIMessage(
            "Response",
            tool_calls=[{"name": "search", "args": {}, "id": "1"}],
        ),
        ToolMessage("Result", tool_call_id="1"),
    ]
    run_messages = _get_run_messages(messages)
    assert len(run_messages) == 2
    assert isinstance(run_messages[0], AIMessage)
    assert isinstance(run_messages[1], ToolMessage)


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
    # Test with end behavior - global tool limit
    middleware = ToolCallLimitMiddleware(thread_limit=3, run_limit=2)

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
                tool_calls=[{"name": "search", "args": {}, "id": "3"}],
            ),
        ],
        "thread_tool_call_count": {"__all__": 3},  # 2 from first AI message + 1 from second
        "run_tool_call_count": {"__all__": 1},  # 1 from second AI message (after HumanMessage)
    }
    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "thread limit (3/3)" in result["messages"][0].content

    # Test when run limit is exceeded
    state = {
        "messages": [
            HumanMessage("Question"),
            AIMessage(
                "Response",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "calculator", "args": {}, "id": "2"},
                ],
            ),
        ],
        "thread_tool_call_count": {"__all__": 2},
        "run_tool_call_count": {"__all__": 2},  # 2 tool calls in current run
    }
    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "run limit (2/2)" in result["messages"][0].content


def test_middleware_with_specific_tool():
    """Test middleware that limits a specific tool."""
    # Limit only the "search" tool
    middleware = ToolCallLimitMiddleware(tool_name="search", thread_limit=2, run_limit=1)

    runtime = None

    # Test with search tool calls
    state = {
        "messages": [
            HumanMessage("Question"),
            AIMessage(
                "Response",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "1"},
                    {"name": "calculator", "args": {}, "id": "2"},
                ],
            ),
        ],
        "thread_tool_call_count": {"search": 1},  # 1 search call
        "run_tool_call_count": {"search": 1},  # 1 search call in current run
    }
    # Run limit for search is 1, should be exceeded
    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert "'search' tool call" in result["messages"][0].content
    assert "run limit (1/1)" in result["messages"][0].content

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
        ],
        "thread_tool_call_count": {"__all__": 2},  # 2 tool calls total
        "run_tool_call_count": {"__all__": 1},  # 1 tool call in current run
    }
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.before_model(state, runtime)  # type: ignore[arg-type]

    assert "thread limit (2/2)" in str(exc_info.value)
    assert exc_info.value.thread_count == 2
    assert exc_info.value.thread_limit == 2

    # Test exception when run limit exceeded
    state = {
        "messages": [
            HumanMessage("Question"),
            AIMessage("R", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ],
        "thread_tool_call_count": {"__all__": 1},
        "run_tool_call_count": {"__all__": 1},  # 1 tool call in current run
    }
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.before_model(state, runtime)  # type: ignore[arg-type]

    assert "run limit (1/1)" in str(exc_info.value)


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
    assert "'search' tool call" in last_message.content
    assert "thread limit" in last_message.content


def test_run_limit_with_multiple_human_messages():
    """Test that run limit correctly resets after each HumanMessage."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    # Model makes 2 tool calls, then 1 tool call, then no tool calls
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

    # Run limit of 2 tool calls per run
    middleware = ToolCallLimitMiddleware(run_limit=2)

    agent = create_agent(
        model=model, tools=[search], middleware=[middleware], checkpointer=InMemorySaver()
    )

    thread_config = {"configurable": {"thread_id": "test_thread"}}

    # First run: 2 tool calls, should hit run limit
    result = agent.invoke({"messages": [HumanMessage("Question 1")]}, thread_config)
    last_message = result["messages"][-1]
    assert "run limit (2/2)" in last_message.content

    # Second run: starts fresh, should also hit run limit after 2 calls
    result = agent.invoke({"messages": [HumanMessage("Question 2")]}, thread_config)
    last_message = result["messages"][-1]
    # This should also hit the limit because run count resets
    assert "run limit (2/2)" in last_message.content


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
        ],
        "thread_tool_call_count": {"__all__": 3},  # 3 tool calls total
        "run_tool_call_count": {"__all__": 2},  # 2 tool calls in current run
    }

    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.before_model(state, None)  # type: ignore[arg-type]

    error_msg = str(exc_info.value)
    assert "Tool call limits exceeded" in error_msg
    assert "thread limit (3/3)" in error_msg

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
        ],
        "thread_tool_call_count": {
            "search": 2
        },  # 2 search calls total (calculator calls don't count)
        "run_tool_call_count": {"search": 1},  # 1 search call in current run
    }

    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        middleware.before_model(state, None)  # type: ignore[arg-type]

    error_msg = str(exc_info.value)
    assert "'search' tool call limits exceeded" in error_msg
    assert "thread limit (2/2)" in error_msg


def test_end_behavior():
    """Test end behavior: stops entire agent when limit hit."""

    @tool
    def tool1(query: str) -> str:
        """First tool."""
        return f"tool1: {query}"

    @tool
    def tool2(query: str) -> str:
        """Second tool."""
        return f"tool2: {query}"

    @tool
    def tool3(query: str) -> str:
        """Third tool."""
        return f"tool3: {query}"

    # Model tries to call all 3 tools
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="tool1", args={"query": "test"}, id="1"),
                ToolCall(name="tool2", args={"query": "test"}, id="2"),
                ToolCall(name="tool3", args={"query": "test"}, id="3"),
            ]
        ]
    )

    # Limit to 2 tool calls with end behavior
    middleware = ToolCallLimitMiddleware(run_limit=2, exit_behavior="end")

    agent = create_agent(
        model=model,
        tools=[tool1, tool2, tool3],
        middleware=[middleware],
    )

    result = agent.invoke({"messages": [HumanMessage("Call all tools")]})

    # Agent should hit limit before calling model, jumping to end
    # The model makes 3 tool calls in first response
    # after_model increments count to 3
    # before_model on next iteration sees count >= limit and jumps to end
    messages = result["messages"]

    # Should have: HumanMessage, AIMessage with tool calls, 3 ToolMessages, AIMessage with limit message
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    # Last message should be the limit exceeded message
    assert (
        "run limit" in ai_messages[-1].content
        or "Tool call limits exceeded" in ai_messages[-1].content
    )


def test_end_tools_behavior():
    """Test end_tools behavior: blocks tool execution but lets agent continue."""

    call_count = {"tool1": 0, "tool2": 0, "tool3": 0}

    @tool
    def tool1(query: str) -> str:
        """First tool."""
        call_count["tool1"] += 1
        return f"tool1 result: {query}"

    @tool
    def tool2(query: str) -> str:
        """Second tool."""
        call_count["tool2"] += 1
        return f"tool2 result: {query}"

    @tool
    def tool3(query: str) -> str:
        """Third tool."""
        call_count["tool3"] += 1
        return f"tool3 result: {query}"

    # Model makes 3 tool calls, then responds with no tool calls
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="tool1", args={"query": "test"}, id="1"),
                ToolCall(name="tool2", args={"query": "test"}, id="2"),
                ToolCall(name="tool3", args={"query": "test"}, id="3"),
            ],
            [],  # Final response with no tool calls
        ]
    )

    # Limit to 2 tool calls with end_tools behavior
    middleware = ToolCallLimitMiddleware(run_limit=2, exit_behavior="end_tools")

    agent = create_agent(
        model=model,
        tools=[tool1, tool2, tool3],
        middleware=[middleware],
    )

    result = agent.invoke({"messages": [HumanMessage("Call all tools")]})

    # Verify that only the first 2 tools were actually executed
    assert call_count["tool1"] == 1, "tool1 should have been called"
    assert call_count["tool2"] == 1, "tool2 should have been called"
    assert call_count["tool3"] == 0, "tool3 should have been blocked"

    # Check messages - should have results from tool1 and tool2, warning from tool3
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 3

    # First two should have actual results
    assert "tool1 result" in tool_messages[0].content
    assert "tool2 result" in tool_messages[1].content

    # Third should be a limit warning
    assert "Tool call limits exceeded" in tool_messages[2].content
    assert "Do not call any more tools" in tool_messages[2].content


def test_error_behavior():
    """Test error behavior: raises exception when limit hit."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    # Model makes 3 tool calls
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "test1"}, id="1"),
                ToolCall(name="search", args={"query": "test2"}, id="2"),
                ToolCall(name="search", args={"query": "test3"}, id="3"),
            ]
        ]
    )

    # Limit to 2 tool calls with error behavior
    middleware = ToolCallLimitMiddleware(run_limit=2, exit_behavior="error")

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[middleware],
    )

    # Should raise exception when limit is hit
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        agent.invoke({"messages": [HumanMessage("Search for things")]})

    assert "run limit" in str(exc_info.value)
    assert exc_info.value.run_limit == 2


def test_end_tools_sequential():
    """Test end_tools behavior with sequential tool calls (one at a time)."""

    call_count = {"search": 0}

    @tool
    def search(query: str) -> str:
        """Search for information."""
        call_count["search"] += 1
        return f"search result {call_count['search']}: {query}"

    # Model makes tool calls one at a time across multiple iterations
    # Each iteration: make one tool call, then continue
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "first"}, id="1")],
            [ToolCall(name="search", args={"query": "second"}, id="2")],
            [ToolCall(name="search", args={"query": "third"}, id="3")],
            [],  # Final response with no tool calls
        ]
    )

    # Limit to 2 tool calls with end_tools behavior
    middleware = ToolCallLimitMiddleware(run_limit=2, exit_behavior="end_tools")

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[middleware],
    )

    result = agent.invoke({"messages": [HumanMessage("Search sequentially")]})

    # First 2 tools should execute, third should be blocked
    assert call_count["search"] == 2, "Only first 2 searches should execute"

    # Check tool messages
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 3

    # First two should have actual results
    assert "search result 1" in tool_messages[0].content
    assert "search result 2" in tool_messages[1].content

    # Third should be blocked
    assert "tool call limits exceeded" in tool_messages[2].content.lower()


def test_end_tools_with_specific_tool():
    """Test end_tools behavior with specific tool limiting."""

    call_count = {"search": 0, "calculator": 0}

    @tool
    def search(query: str) -> str:
        """Search for information."""
        call_count["search"] += 1
        return f"search result: {query}"

    @tool
    def calculator(expression: str) -> str:
        """Calculate an expression."""
        call_count["calculator"] += 1
        return f"calculator result: {expression}"

    # Model makes 3 search calls and 2 calculator calls
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "test1"}, id="1"),
                ToolCall(name="search", args={"query": "test2"}, id="2"),
                ToolCall(name="calculator", args={"expression": "1+1"}, id="3"),
                ToolCall(name="search", args={"query": "test3"}, id="4"),
                ToolCall(name="calculator", args={"expression": "2+2"}, id="5"),
            ],
            [],
        ]
    )

    # Limit only search to 2 calls with end_tools behavior
    middleware = ToolCallLimitMiddleware(tool_name="search", run_limit=2, exit_behavior="end_tools")

    agent = create_agent(
        model=model,
        tools=[search, calculator],
        middleware=[middleware],
    )

    result = agent.invoke({"messages": [HumanMessage("Search and calculate")]})

    # First 2 search calls should execute, third should be blocked
    assert call_count["search"] == 2, "Only first 2 search calls should execute"
    # All calculator calls should execute (not limited)
    assert call_count["calculator"] == 2, "All calculator calls should execute"

    # Check tool messages
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    search_messages = [m for m in tool_messages if m.name == "search"]
    calc_messages = [m for m in tool_messages if m.name == "calculator"]

    # Should have 3 search messages (2 results + 1 warning)
    assert len(search_messages) == 3
    assert "search result" in search_messages[0].content
    assert "search result" in search_messages[1].content
    assert "tool call limits exceeded" in search_messages[2].content.lower()

    # Should have 2 calculator results
    assert len(calc_messages) == 2
    assert "calculator result" in calc_messages[0].content
    assert "calculator result" in calc_messages[1].content


def test_end_tools_thread_limit():
    """Test end_tools behavior with thread limit across multiple runs."""

    call_count = {"search": 0}

    @tool
    def search(query: str) -> str:
        """Search for information."""
        call_count["search"] += 1
        return f"search result {call_count['search']}: {query}"

    # First run: 2 search calls
    # Second run: 2 more search calls (total 4)
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "test1"}, id="1"),
                ToolCall(name="search", args={"query": "test2"}, id="2"),
            ],
            [],
            [
                ToolCall(name="search", args={"query": "test3"}, id="3"),
                ToolCall(name="search", args={"query": "test4"}, id="4"),
            ],
            [],
        ]
    )

    # Thread limit of 3 with end_tools behavior
    middleware = ToolCallLimitMiddleware(thread_limit=3, exit_behavior="end_tools")

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )

    thread_config = {"configurable": {"thread_id": "test_thread"}}

    # First run: 2 calls, both should execute
    result1 = agent.invoke({"messages": [HumanMessage("First search")]}, thread_config)
    assert call_count["search"] == 2

    tool_messages_1 = [m for m in result1["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages_1) == 2
    assert "search result" in tool_messages_1[0].content
    assert "search result" in tool_messages_1[1].content

    # Second run: would add 2 more (total 4), but limit is 3
    # So only first call should execute, second should be blocked
    result2 = agent.invoke({"messages": [HumanMessage("Second search")]}, thread_config)
    assert call_count["search"] == 3, "Only 1 more search should execute (3 total)"

    tool_messages_2 = [
        m for m in result2["messages"] if isinstance(m, ToolMessage) and m not in tool_messages_1
    ]

    # Should have 2 new tool messages: 1 result, 1 warning
    assert len(tool_messages_2) == 2
    # First is the successful execution
    assert "search result 3" in tool_messages_2[0].content
    # Second is the blocked execution
    assert "Tool call limits exceeded" in tool_messages_2[1].content


def test_comparison_all_three_behaviors():
    """
    Comprehensive test comparing all three exit behaviors.

    This demonstrates the key differences:
    - end: Stops entire agent execution
    - end_tools: Blocks individual tool executions, agent continues
    - error: Raises exception
    """

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"search: {query}"

    # Model tries to make 3 tool calls
    def make_model():
        return FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(name="search", args={"query": "q1"}, id="1"),
                    ToolCall(name="search", args={"query": "q2"}, id="2"),
                    ToolCall(name="search", args={"query": "q3"}, id="3"),
                ],
                [],
            ]
        )

    # Test 1: end behavior
    agent_end = create_agent(
        model=make_model(),
        tools=[search],
        middleware=[ToolCallLimitMiddleware(run_limit=2, exit_behavior="end")],
    )

    result_end = agent_end.invoke({"messages": [HumanMessage("Search")]})

    # With end: agent completes first iteration (3 tool calls),
    # then before_model on next iteration sees limit exceeded and jumps to end
    messages_end = result_end["messages"]
    ai_messages_end = [m for m in messages_end if isinstance(m, AIMessage)]

    # Should have limit exceeded message
    assert any("limit" in m.content.lower() for m in ai_messages_end)

    # Test 2: end_tools behavior
    agent_end_tools = create_agent(
        model=make_model(),
        tools=[search],
        middleware=[ToolCallLimitMiddleware(run_limit=2, exit_behavior="end_tools")],
    )

    result_end_tools = agent_end_tools.invoke({"messages": [HumanMessage("Search")]})

    # With end_tools: first 2 tools execute, third gets warning, agent continues
    tool_messages_end_tools = [
        m for m in result_end_tools["messages"] if isinstance(m, ToolMessage)
    ]
    assert len(tool_messages_end_tools) == 3
    assert "search: q1" in tool_messages_end_tools[0].content
    assert "search: q2" in tool_messages_end_tools[1].content
    assert "Tool call limits exceeded" in tool_messages_end_tools[2].content

    # Test 3: error behavior
    agent_error = create_agent(
        model=make_model(),
        tools=[search],
        middleware=[ToolCallLimitMiddleware(run_limit=2, exit_behavior="error")],
    )

    # With error: should raise exception
    with pytest.raises(ToolCallLimitExceededError) as exc_info:
        agent_error.invoke({"messages": [HumanMessage("Search")]})

    assert exc_info.value.run_limit == 2
