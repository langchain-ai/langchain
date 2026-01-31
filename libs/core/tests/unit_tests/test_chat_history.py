"""Test InMemoryChatMessageHistory serialization.

Tests to verify that subclass-specific fields like tool_calls on AIMessage
are preserved during Pydantic serialization/deserialization.
"""

import json

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import (
    invalid_tool_call as create_invalid_tool_call,
)
from langchain_core.messages.tool import (
    tool_call as create_tool_call,
)


def test_inmemory_chat_history_preserves_tool_calls_model_dump() -> None:
    """Test that tool_calls are preserved after model_dump() serialization."""
    history = InMemoryChatMessageHistory()

    ai_msg = AIMessage(
        content="I'll help you with that.",
        tool_calls=[create_tool_call(name="get_weather", args={"city": "NYC"}, id="1")],
    )
    history.add_message(ai_msg)

    # Serialize using model_dump
    dumped = history.model_dump()

    # Verify tool_calls are present in the serialized data
    assert len(dumped["messages"]) == 1
    assert "tool_calls" in dumped["messages"][0]
    assert len(dumped["messages"][0]["tool_calls"]) == 1
    assert dumped["messages"][0]["tool_calls"][0]["name"] == "get_weather"
    assert dumped["messages"][0]["tool_calls"][0]["args"] == {"city": "NYC"}
    assert dumped["messages"][0]["tool_calls"][0]["id"] == "1"


def test_inmemory_chat_history_preserves_tool_calls_model_dump_json() -> None:
    """Test that tool_calls are preserved after model_dump_json() serialization."""
    history = InMemoryChatMessageHistory()

    ai_msg = AIMessage(
        content="Let me search for that.",
        tool_calls=[
            create_tool_call(name="search", args={"query": "langchain"}, id="abc123")
        ],
    )
    history.add_message(ai_msg)

    # Serialize to JSON string
    json_str = history.model_dump_json()
    parsed = json.loads(json_str)

    # Verify tool_calls are present in the JSON
    assert len(parsed["messages"]) == 1
    assert "tool_calls" in parsed["messages"][0]
    assert len(parsed["messages"][0]["tool_calls"]) == 1
    assert parsed["messages"][0]["tool_calls"][0]["name"] == "search"
    assert parsed["messages"][0]["tool_calls"][0]["args"] == {"query": "langchain"}


def test_inmemory_chat_history_preserves_multiple_tool_calls() -> None:
    """Test that multiple tool_calls are preserved during serialization."""
    history = InMemoryChatMessageHistory()

    ai_msg = AIMessage(
        content="I'll get the weather and search for restaurants.",
        tool_calls=[
            create_tool_call(name="get_weather", args={"city": "NYC"}, id="1"),
            create_tool_call(
                name="search_restaurants", args={"cuisine": "italian"}, id="2"
            ),
        ],
    )
    history.add_message(ai_msg)

    dumped = history.model_dump()

    assert len(dumped["messages"][0]["tool_calls"]) == 2
    assert dumped["messages"][0]["tool_calls"][0]["name"] == "get_weather"
    assert dumped["messages"][0]["tool_calls"][1]["name"] == "search_restaurants"


def test_inmemory_chat_history_serializes_mixed_message_types() -> None:
    """Test that mixed message types with tool_calls serialize correctly."""
    history = InMemoryChatMessageHistory()

    # Add messages of different types
    history.add_message(SystemMessage(content="You are a helpful assistant."))
    history.add_message(HumanMessage(content="What's the weather in NYC?"))
    history.add_message(
        AIMessage(
            content="Let me check that for you.",
            tool_calls=[
                create_tool_call(name="get_weather", args={"city": "NYC"}, id="tool_1")
            ],
        )
    )

    # Serialize to JSON
    json_str = history.model_dump_json()
    dumped = json.loads(json_str)

    # Verify all messages are serialized with their type field
    assert len(dumped["messages"]) == 3
    assert dumped["messages"][0]["type"] == "system"
    assert dumped["messages"][1]["type"] == "human"
    assert dumped["messages"][2]["type"] == "ai"

    # Verify tool_calls are preserved in the AI message
    assert "tool_calls" in dumped["messages"][2]
    assert len(dumped["messages"][2]["tool_calls"]) == 1
    assert dumped["messages"][2]["tool_calls"][0]["name"] == "get_weather"
    assert dumped["messages"][2]["tool_calls"][0]["args"] == {"city": "NYC"}


def test_inmemory_chat_history_preserves_invalid_tool_calls() -> None:
    """Test that invalid_tool_calls are also preserved during serialization."""
    history = InMemoryChatMessageHistory()

    ai_msg = AIMessage(
        content="",
        tool_calls=[create_tool_call(name="valid_tool", args={"key": "value"}, id="1")],
        invalid_tool_calls=[
            create_invalid_tool_call(
                name="invalid_tool", args="not valid json", id="2", error="parse error"
            )
        ],
    )
    history.add_message(ai_msg)

    dumped = history.model_dump()

    assert len(dumped["messages"][0]["tool_calls"]) == 1
    assert len(dumped["messages"][0]["invalid_tool_calls"]) == 1
    assert dumped["messages"][0]["invalid_tool_calls"][0]["name"] == "invalid_tool"
    assert dumped["messages"][0]["invalid_tool_calls"][0]["error"] == "parse error"


def test_inmemory_chat_history_empty_tool_calls() -> None:
    """Test that AIMessage without tool_calls serializes correctly."""
    history = InMemoryChatMessageHistory()

    ai_msg = AIMessage(content="Hello! How can I help you today?")
    history.add_message(ai_msg)

    dumped = history.model_dump()

    # Should have empty tool_calls list
    assert dumped["messages"][0]["tool_calls"] == []
    assert dumped["messages"][0]["content"] == "Hello! How can I help you today?"
