"""Tests for invalid_tool_calls handling in create_agent.

Verifies that agents do not silently terminate when the LLM produces malformed
JSON for tool calls, and instead surface the error back to the LLM for retry.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import InvalidToolCall
from langgraph.errors import GraphRecursionError

from langchain.agents import create_agent
from langchain.tools import tool as dec_tool
from tests.unit_tests.agents.model import FakeToolCallingModel


@dec_tool
def get_weather(city: str) -> str:
    """Get weather for a city.

    Args:
        city: The city name.
    """
    return f"Sunny in {city}"


class TestInvalidToolCallRetry:
    """Tests that invalid_tool_calls trigger error feedback and retry."""

    def test_invalid_tool_call_retries_and_succeeds(self) -> None:
        """Agent retries after invalid tool call and eventually succeeds.

        Flow:
        1. Model emits invalid_tool_calls (malformed JSON)
        2. Agent converts to error ToolMessage and routes back to model
        3. Model retries with valid tool_call
        4. Tool executes successfully
        5. Model returns final text answer
        """
        model = FakeToolCallingModel(
            tool_calls=[
                [],  # Round 0: no valid tool_calls (only invalid)
                [{"name": "get_weather", "args": {"city": "SF"}, "id": "call_2"}],
                [],  # Round 2: final text response
            ],
            invalid_tool_calls=[
                [
                    InvalidToolCall(
                        type="invalid_tool_call",
                        name="get_weather",
                        args='{city: "SF"}',
                        id="call_1",
                        error="JSON parse error",
                    )
                ],
                [],  # Round 1: no invalid calls
                [],  # Round 2: no invalid calls
            ],
        )

        agent = create_agent(model, tools=[get_weather])
        result = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        messages = result["messages"]

        # Model should have been called more than once
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 2, "Model should be called at least twice (initial + retry)"

        # There should be an error ToolMessage from the invalid tool call
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        error_tool_messages = [m for m in tool_messages if m.status == "error"]
        assert len(error_tool_messages) >= 1, (
            "At least one error ToolMessage should exist for the invalid tool call"
        )

        # The error message should reference the tool name
        assert any("get_weather" in m.content for m in error_tool_messages), (
            "Error ToolMessage should mention the tool name"
        )

        # There should also be a successful tool execution
        success_tool_messages = [m for m in tool_messages if m.status == "success"]
        assert len(success_tool_messages) >= 1, (
            "At least one successful ToolMessage should exist from the retry"
        )

    def test_multiple_invalid_tool_calls_all_get_error_messages(self) -> None:
        """Each invalid tool call produces its own error ToolMessage."""
        model = FakeToolCallingModel(
            tool_calls=[
                [],  # Round 0: no valid calls
                [],  # Round 1: final response (end loop)
            ],
            invalid_tool_calls=[
                [
                    InvalidToolCall(
                        type="invalid_tool_call",
                        name="get_weather",
                        args="{bad json 1",
                        id="call_1",
                        error="parse error 1",
                    ),
                    InvalidToolCall(
                        type="invalid_tool_call",
                        name="get_weather",
                        args="{bad json 2",
                        id="call_2",
                        error="parse error 2",
                    ),
                ],
                [],  # Round 1: no invalid calls
            ],
        )

        agent = create_agent(model, tools=[get_weather])
        result = agent.invoke({"messages": [HumanMessage("Weather?")]})

        messages = result["messages"]
        error_tool_messages = [
            m for m in messages if isinstance(m, ToolMessage) and m.status == "error"
        ]
        assert len(error_tool_messages) == 2, (
            "Each invalid tool call should produce its own error ToolMessage"
        )

    def test_invalid_tool_call_with_none_id(self) -> None:
        """InvalidToolCall with id=None does not crash the agent."""
        model = FakeToolCallingModel(
            tool_calls=[
                [],  # Round 0: only invalid
                [],  # Round 1: end loop
            ],
            invalid_tool_calls=[
                [
                    InvalidToolCall(
                        type="invalid_tool_call",
                        name="get_weather",
                        args="{bad",
                        id=None,
                        error=None,
                    )
                ],
                [],
            ],
        )

        agent = create_agent(model, tools=[get_weather])
        result = agent.invoke({"messages": [HumanMessage("Weather?")]})

        error_messages = [
            m for m in result["messages"] if isinstance(m, ToolMessage) and m.status == "error"
        ]
        assert len(error_messages) == 1
        # Should not crash; tool_call_id defaults to ""
        assert error_messages[0].tool_call_id == ""

    def test_invalid_tool_call_with_none_name(self) -> None:
        """InvalidToolCall with name=None does not crash the agent."""
        model = FakeToolCallingModel(
            tool_calls=[
                [],  # Round 0: only invalid
                [],  # Round 1: end loop
            ],
            invalid_tool_calls=[
                [
                    InvalidToolCall(
                        type="invalid_tool_call",
                        name=None,
                        args="{bad",
                        id="call_1",
                        error=None,
                    )
                ],
                [],
            ],
        )

        agent = create_agent(model, tools=[get_weather])
        result = agent.invoke({"messages": [HumanMessage("Weather?")]})

        error_messages = [
            m for m in result["messages"] if isinstance(m, ToolMessage) and m.status == "error"
        ]
        assert len(error_messages) == 1
        assert error_messages[0].name == "unknown"


class TestInvalidToolCallBackwardCompat:
    """Tests that existing behavior is preserved."""

    def test_valid_tool_calls_unchanged(self) -> None:
        """Normal valid tool calls work exactly as before."""
        model = FakeToolCallingModel(
            tool_calls=[
                [{"name": "get_weather", "args": {"city": "NYC"}, "id": "call_1"}],
                [],  # End loop
            ],
        )

        agent = create_agent(model, tools=[get_weather])
        result = agent.invoke({"messages": [HumanMessage("Weather in NYC?")]})

        messages = result["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].status == "success"
        assert "Sunny in NYC" in tool_messages[0].content

    def test_no_tool_calls_exits_normally(self) -> None:
        """Agent exits normally when model produces no tool calls at all."""
        model = FakeToolCallingModel(
            tool_calls=[[]],  # No tool calls, should exit
        )

        agent = create_agent(model, tools=[get_weather])
        result = agent.invoke({"messages": [HumanMessage("Hello")]})

        messages = result["messages"]
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        # Should exit after single model call with no tool calls
        assert len(ai_messages) == 1
        assert len(ai_messages[0].tool_calls) == 0
        assert len(ai_messages[0].invalid_tool_calls) == 0


class TestInvalidToolCallRecursionLimit:
    """Tests that persistent invalid tool calls hit the recursion limit."""

    def test_persistent_invalid_calls_hit_recursion_limit(self) -> None:
        """If model always produces invalid tool calls, recursion limit is hit."""
        # Model always returns invalid tool calls (never corrects itself)
        model = FakeToolCallingModel(
            tool_calls=[[]],  # Always empty valid calls (cycles)
            invalid_tool_calls=[
                [
                    InvalidToolCall(
                        type="invalid_tool_call",
                        name="get_weather",
                        args="{always bad",
                        id="call_loop",
                        error="persistent parse error",
                    )
                ]
            ],  # Always invalid (cycles)
        )

        agent = create_agent(model, tools=[get_weather])

        with pytest.raises(GraphRecursionError):
            agent.invoke(
                {"messages": [HumanMessage("Weather?")]},
                {"recursion_limit": 5},
            )
