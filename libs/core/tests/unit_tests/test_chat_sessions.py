"""Tests for chat_sessions module."""

from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import AIMessage, HumanMessage


def test_chat_session_typeddict():
    """Test that ChatSession TypedDict works correctly."""
    # Test with minimal fields
    session1: ChatSession = {"messages": []}
    assert session1["messages"] == []

    # Test with messages
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there")]
    session2: ChatSession = {"messages": messages}
    assert len(session2["messages"]) == 2
    assert session2["messages"][0].content == "Hello"
    assert session2["messages"][1].content == "Hi there"

    # Test with functions
    functions = [{"name": "test_func", "description": "A test function"}]
    session3: ChatSession = {"messages": messages, "functions": functions}
    assert session3["messages"] == messages
    assert session3["functions"] == functions

    # Test empty ChatSession (total=False allows missing fields)
    session4: ChatSession = {}
    assert "messages" not in session4
    assert "functions" not in session4


def test_chat_session_is_typeddict():
    """Test that ChatSession is a TypedDict."""
    from typing import get_origin, get_type_hints

    # Check that it's a TypedDict type
    origin = get_origin(ChatSession)
    assert origin is not None

    # Check type hints
    hints = get_type_hints(ChatSession)
    assert "messages" in hints
    assert "functions" in hints
