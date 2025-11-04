"""Tests for parallel tool call handling in ToolCallLimitMiddleware."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.factory import create_agent
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel


def test_parallel_tool_calls_with_limit_continue_mode():
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
        assert "limit" in error_msg.content.lower()

    # Verify execution continued (no early termination)
    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
    # Should have: initial AI message with 3 tool calls, then final AI message (no tool calls)
    assert len(ai_messages) >= 2


def test_parallel_tool_calls_with_limit_end_mode():
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

    # Verify error messages explain the limit
    for error_msg in error_tool_messages:
        assert "limit" in error_msg.content.lower()

    # Verify AI message explaining why execution stopped
    ai_limit_messages = [
        msg
        for msg in messages
        if isinstance(msg, AIMessage) and "limit" in msg.content.lower() and not msg.tool_calls
    ]
    assert len(ai_limit_messages) == 1, "Should have exactly one AI message explaining the limit"

    ai_msg_content = ai_limit_messages[0].content.lower()
    assert "do not" in ai_msg_content or "don't" in ai_msg_content, (
        "Should instruct model not to call tool again"
    )


def test_parallel_mixed_tool_calls_with_specific_tool_limit():
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

    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    search_success = [m for m in tool_messages if "Search:" in m.content]
    search_blocked = [
        m for m in tool_messages if "limit" in m.content.lower() and "search" in m.content.lower()
    ]
    calc_success = [m for m in tool_messages if "Calc:" in m.content]

    assert len(search_success) == 1, "Should have 1 successful search call"
    assert len(search_blocked) == 2, "Should have 2 blocked search calls"
    assert len(calc_success) == 2, "All calculator calls should succeed (not limited)"
