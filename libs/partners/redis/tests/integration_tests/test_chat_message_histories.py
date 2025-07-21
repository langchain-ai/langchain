"""Integration tests for RedisChatMessageHistory.

These tests require a running Redis instance. 
Set REDIS_URL environment variable or skip these tests.
"""

import os
import pytest
from typing import Generator

from langchain_core.messages import AIMessage, HumanMessage

from langchain_redis import RedisChatMessageHistory


def test_redis_chat_message_history_integration() -> None:
    """Test RedisChatMessageHistory with a real Redis instance."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Skip test if Redis is not available
    try:
        import redis
        client = redis.from_url(redis_url)
        client.ping()
    except Exception:
        pytest.skip("Redis server not available")
    
    # Test session IDs
    session_id = "integration_test_session"
    key_prefix = "integration_test:"
    
    # Create history with key_prefix
    history = RedisChatMessageHistory(
        session_id=session_id,
        redis_url=redis_url,
        key_prefix=key_prefix
    )
    
    # Clear any existing messages
    history.clear()
    
    # Initially should have no messages
    assert history.messages == []
    
    # Add messages
    history.add_user_message("Hello, AI assistant!")
    history.add_ai_message("Hello! How can I assist you today?")
    
    # Retrieve messages
    messages = history.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello, AI assistant!"
    assert isinstance(messages[1], AIMessage) 
    assert messages[1].content == "Hello! How can I assist you today?"
    
    # Verify key is constructed correctly
    expected_key = f"{key_prefix}{session_id}"
    assert history.key == expected_key
    
    # Test that another instance with same session_id can retrieve messages
    history2 = RedisChatMessageHistory(
        session_id=session_id,
        redis_url=redis_url,
        key_prefix=key_prefix
    )
    
    messages2 = history2.messages
    assert len(messages2) == 2
    assert messages2[0].content == "Hello, AI assistant!"
    assert messages2[1].content == "Hello! How can I assist you today?"
    
    # Clean up
    history.clear()
    assert history.messages == []


def test_redis_chat_message_history_without_key_prefix() -> None:
    """Test RedisChatMessageHistory without key_prefix."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Skip test if Redis is not available
    try:
        import redis
        client = redis.from_url(redis_url)
        client.ping()
    except Exception:
        pytest.skip("Redis server not available")
    
    session_id = "integration_test_no_prefix"
    
    # Create history without key_prefix
    history = RedisChatMessageHistory(
        session_id=session_id,
        redis_url=redis_url
    )
    
    # Clear any existing messages
    history.clear()
    
    # Add and retrieve messages
    history.add_user_message("Test message")
    messages = history.messages
    
    assert len(messages) == 1
    assert messages[0].content == "Test message"
    assert history.key == session_id
    
    # Clean up
    history.clear()


def test_redis_chat_message_history_with_ttl() -> None:
    """Test RedisChatMessageHistory with TTL."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Skip test if Redis is not available
    try:
        import redis
        client = redis.from_url(redis_url)
        client.ping()
    except Exception:
        pytest.skip("Redis server not available")
    
    session_id = "integration_test_ttl"
    key_prefix = "ttl_test:"
    
    # Create history with TTL
    history = RedisChatMessageHistory(
        session_id=session_id,
        redis_url=redis_url,
        key_prefix=key_prefix,
        ttl=60  # 60 seconds
    )
    
    # Clear and add messages
    history.clear()
    history.add_user_message("TTL test message")
    
    # Verify message is there
    messages = history.messages
    assert len(messages) == 1
    assert messages[0].content == "TTL test message"
    
    # Verify TTL is set in Redis
    redis_client = redis.from_url(redis_url)
    ttl = redis_client.ttl(history.key)
    assert ttl > 0 and ttl <= 60  # Should be set and less than or equal to 60
    
    # Clean up
    history.clear()