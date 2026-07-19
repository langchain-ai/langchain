"""Tests for DeduplicateToolCallsMiddleware functionality."""

from langchain_core.messages import AIMessage, HumanMessage, ToolCall, InvalidToolCall
from langchain_core.tools import tool

from langchain.agents.factory import create_agent
from langchain.agents.middleware import DeduplicateToolCallsMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def dummy_tool(query: str, limit: int = 10) -> str:
    """A dummy tool for testing."""
    return f"results for {query} with limit {limit}"


def test_deduplicate_tool_calls_sync() -> None:
    """Test sync deduplication of parallel tool calls."""
    tool_calls = [
        [
            ToolCall(name="dummy_tool", args={"query": "test", "limit": 10}, id="call_1"),
            ToolCall(name="dummy_tool", args={"limit": 10, "query": "test"}, id="call_2"),  # Duplicate (reordered keys)
            ToolCall(name="dummy_tool", args={"query": "test", "limit": 10}, id="call_3"),  # Duplicate (exact)
            ToolCall(name="dummy_tool", args={"query": "other", "limit": 10}, id="call_4"),  # Distinct (different query)
            ToolCall(name="dummy_tool", args={"query": "test", "limit": 5}, id="call_5"),   # Distinct (different limit)
        ],
        [],
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)
    agent = create_agent(
        model=model,
        tools=[dummy_tool],
        middleware=[DeduplicateToolCallsMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("search")]})

    # Find the AI message containing the tool calls
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1

    # The first AIMessage generated will contain the tool calls
    deduplicated_calls = ai_messages[0].tool_calls
    assert len(deduplicated_calls) == 3
    assert deduplicated_calls[0]["id"] == "call_1"
    assert deduplicated_calls[1]["id"] == "call_4"
    assert deduplicated_calls[2]["id"] == "call_5"


async def test_deduplicate_tool_calls_async() -> None:
    """Test async deduplication of parallel tool calls."""
    tool_calls = [
        [
            ToolCall(name="dummy_tool", args={"query": "test"}, id="call_1"),
            ToolCall(name="dummy_tool", args={"query": "test"}, id="call_2"),
        ],
        [],
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)
    agent = create_agent(
        model=model,
        tools=[dummy_tool],
        middleware=[DeduplicateToolCallsMiddleware()],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("search")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1

    deduplicated_calls = ai_messages[0].tool_calls
    assert len(deduplicated_calls) == 1
    assert deduplicated_calls[0]["id"] == "call_1"


def test_deduplicate_invalid_tool_calls() -> None:
    """Test deduplication of invalid tool calls."""
    msg = AIMessage(
        content="",
        tool_calls=[],
        invalid_tool_calls=[
            InvalidToolCall(name="dummy_tool", args="bad json", id="call_1", error="parsing error"),
            InvalidToolCall(name="dummy_tool", args="bad json", id="call_2", error="parsing error"),
        ]
    )

    middleware = DeduplicateToolCallsMiddleware()
    deduplicated_msg = middleware._deduplicate_message(msg)

    assert isinstance(deduplicated_msg, AIMessage)
    assert len(deduplicated_msg.invalid_tool_calls) == 1
    assert deduplicated_msg.invalid_tool_calls[0]["id"] == "call_1"
