"""Test agent name parameter in create_agent.

This module tests that the name parameter correctly sets .name on AIMessage outputs.
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolCall,
)
from langchain_core.tools import tool

from langchain.agents import create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def simple_tool(x: int) -> str:
    """Simple tool for basic tests."""
    return f"Result: {x}"


def test_agent_name_set_on_ai_message() -> None:
    """Test that agent name is set on AIMessage when name is provided."""
    tool_calls: list[list[ToolCall]] = [[]]
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=tool_calls),
        name="test_agent",
    )

    result = agent.invoke({"messages": [HumanMessage("Hello")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1
    assert ai_messages[0].name == "test_agent"


def test_agent_name_not_set_when_none() -> None:
    """Test that AIMessage.name is not set when name is not provided."""
    tool_calls: list[list[ToolCall]] = [[]]
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=tool_calls),
    )

    result = agent.invoke({"messages": [HumanMessage("Hello")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1
    assert ai_messages[0].name is None


def test_agent_name_on_multiple_iterations() -> None:
    """Test that agent name is set on all AIMessages in multi-turn conversation."""
    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 1}, "id": "call_1", "name": "simple_tool"}], []]
        ),
        tools=[simple_tool],
        name="multi_turn_agent",
    )

    result = agent.invoke({"messages": [HumanMessage("Call a tool")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 2
    for msg in ai_messages:
        assert msg.name == "multi_turn_agent"


async def test_agent_name_async() -> None:
    """Test that agent name is set on AIMessage in async execution."""
    tool_calls: list[list[ToolCall]] = [[]]
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=tool_calls),
        name="async_agent",
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Hello async")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1
    assert ai_messages[0].name == "async_agent"


async def test_agent_name_async_multiple_iterations() -> None:
    """Test that agent name is set on all AIMessages in async multi-turn."""
    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 5}, "id": "call_1", "name": "simple_tool"}], []]
        ),
        tools=[simple_tool],
        name="async_multi_agent",
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Call tool async")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 2
    for msg in ai_messages:
        assert msg.name == "async_multi_agent"


# Tests for lc_agent_name in streaming metadata


def test_lc_agent_name_in_stream_metadata() -> None:
    """Test that lc_agent_name is included in metadata when streaming with name."""
    tool_calls: list[list[ToolCall]] = [[]]
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=tool_calls),
        name="streaming_agent",
    )

    metadata_with_agent_name = []
    for _chunk, metadata in agent.stream(
        {"messages": [HumanMessage("Hello")]},
        stream_mode="messages",
    ):
        if "lc_agent_name" in metadata:
            metadata_with_agent_name.append(metadata["lc_agent_name"])

    assert len(metadata_with_agent_name) > 0
    assert all(name == "streaming_agent" for name in metadata_with_agent_name)


def test_lc_agent_name_not_in_stream_metadata_when_name_not_provided() -> None:
    """Test that lc_agent_name is not in metadata when name is not provided."""
    tool_calls: list[list[ToolCall]] = [[]]
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=tool_calls),
    )

    for _chunk, metadata in agent.stream(
        {"messages": [HumanMessage("Hello")]},
        stream_mode="messages",
    ):
        assert "lc_agent_name" not in metadata


def test_lc_agent_name_in_stream_metadata_multiple_iterations() -> None:
    """Test that lc_agent_name is in metadata for all stream events in multi-turn."""
    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 1}, "id": "call_1", "name": "simple_tool"}], []]
        ),
        tools=[simple_tool],
        name="multi_turn_streaming_agent",
    )

    metadata_with_agent_name = []
    for _chunk, metadata in agent.stream(
        {"messages": [HumanMessage("Call a tool")]},
        stream_mode="messages",
    ):
        if "lc_agent_name" in metadata:
            metadata_with_agent_name.append(metadata["lc_agent_name"])

    # Should have metadata entries for messages from both iterations
    assert len(metadata_with_agent_name) > 0
    assert all(name == "multi_turn_streaming_agent" for name in metadata_with_agent_name)


async def test_lc_agent_name_in_astream_metadata() -> None:
    """Test that lc_agent_name is included in metadata when async streaming with name."""
    tool_calls: list[list[ToolCall]] = [[]]
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=tool_calls),
        name="async_streaming_agent",
    )

    metadata_with_agent_name = []
    async for _chunk, metadata in agent.astream(
        {"messages": [HumanMessage("Hello async")]},
        stream_mode="messages",
    ):
        if "lc_agent_name" in metadata:
            metadata_with_agent_name.append(metadata["lc_agent_name"])

    assert len(metadata_with_agent_name) > 0
    assert all(name == "async_streaming_agent" for name in metadata_with_agent_name)


async def test_lc_agent_name_not_in_astream_metadata_when_name_not_provided() -> None:
    """Test that lc_agent_name is not in async stream metadata when name not provided."""
    tool_calls: list[list[ToolCall]] = [[]]
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=tool_calls),
    )

    async for _chunk, metadata in agent.astream(
        {"messages": [HumanMessage("Hello async")]},
        stream_mode="messages",
    ):
        assert "lc_agent_name" not in metadata


async def test_lc_agent_name_in_astream_metadata_multiple_iterations() -> None:
    """Test that lc_agent_name is in metadata for all async stream events in multi-turn."""
    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 5}, "id": "call_1", "name": "simple_tool"}], []]
        ),
        tools=[simple_tool],
        name="async_multi_turn_streaming_agent",
    )

    metadata_with_agent_name = []
    async for _chunk, metadata in agent.astream(
        {"messages": [HumanMessage("Call tool async")]},
        stream_mode="messages",
    ):
        if "lc_agent_name" in metadata:
            metadata_with_agent_name.append(metadata["lc_agent_name"])

    # Should have metadata entries for messages from both iterations
    assert len(metadata_with_agent_name) > 0
    assert all(name == "async_multi_turn_streaming_agent" for name in metadata_with_agent_name)
