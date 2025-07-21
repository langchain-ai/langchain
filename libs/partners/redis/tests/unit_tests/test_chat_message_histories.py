"""Test RedisChatMessageHistory functionality."""

import json
import pytest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from langchain_redis import RedisChatMessageHistory


class TestRedisChatMessageHistory:
    """Test cases for RedisChatMessageHistory."""

    def test_initialization_with_redis_url(self) -> None:
        """Test initialization with redis_url."""
        with patch("redis.from_url") as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            
            history = RedisChatMessageHistory(
                session_id="test_session",
                redis_url="redis://localhost:6379"
            )
            
            assert history.session_id == "test_session"
            assert history.key_prefix == ""
            assert history._key == "test_session"
            assert history.redis_client == mock_client
            mock_redis.assert_called_once_with("redis://localhost:6379")

    def test_initialization_with_key_prefix(self) -> None:
        """Test initialization with key_prefix."""
        with patch("redis.from_url") as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            
            history = RedisChatMessageHistory(
                session_id="test_session",
                redis_url="redis://localhost:6379",
                key_prefix="chat_app:"
            )
            
            assert history.session_id == "test_session"
            assert history.key_prefix == "chat_app:"
            assert history._key == "chat_app:test_session"
            assert history.redis_client == mock_client

    def test_initialization_with_redis_client(self) -> None:
        """Test initialization with provided redis_client."""
        mock_client = Mock()
        
        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_client
        )
        
        assert history.session_id == "test_session"
        assert history.key_prefix == ""
        assert history._key == "test_session"
        assert history.redis_client == mock_client

    def test_initialization_without_redis_url_or_client(self) -> None:
        """Test that initialization raises error without redis_url or redis_client."""
        with pytest.raises(ValueError, match="Either redis_url or redis_client must be provided"):
            RedisChatMessageHistory(session_id="test_session")

    def test_add_messages_and_retrieve_without_key_prefix(self) -> None:
        """Test adding and retrieving messages without key_prefix."""
        mock_client = Mock()
        
        # Mock stored messages in Redis
        stored_data = [
            {"type": "human", "data": {"content": "Hello, AI assistant!"}},
            {"type": "ai", "data": {"content": "Hello! How can I assist you today?"}}
        ]
        mock_client.get.return_value = json.dumps(stored_data).encode('utf-8')
        mock_client.set.return_value = None
        
        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_client
        )
        
        # Add messages
        history.add_user_message("Hello, AI assistant!")
        history.add_ai_message("Hello! How can I assist you today!")
        
        # Verify messages are retrieved correctly
        messages = history.messages
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello, AI assistant!"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hello! How can I assist you today?"
        
        # Verify Redis key used correctly
        mock_client.get.assert_called_with("test_session")

    def test_add_messages_and_retrieve_with_key_prefix(self) -> None:
        """Test adding and retrieving messages with key_prefix."""
        mock_client = Mock()
        
        # Mock stored messages in Redis
        stored_data = [
            {"type": "human", "data": {"content": "Hello, AI assistant!"}},
            {"type": "ai", "data": {"content": "Hello! How can I assist you today?"}}
        ]
        mock_client.get.return_value = json.dumps(stored_data).encode('utf-8')
        mock_client.set.return_value = None
        
        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_client,
            key_prefix="chat_test:"
        )
        
        # Add messages
        history.add_user_message("Hello, AI assistant!")
        history.add_ai_message("Hello! How can I assist you today!")
        
        # Verify messages are retrieved correctly
        messages = history.messages
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello, AI assistant!"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hello! How can I assist you today?"
        
        # Verify Redis key used correctly with prefix
        mock_client.get.assert_called_with("chat_test:test_session")

    def test_messages_property_empty_when_no_data(self) -> None:
        """Test that messages property returns empty list when no data in Redis."""
        mock_client = Mock()
        mock_client.get.return_value = None  # No data in Redis
        
        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_client,
            key_prefix="chat_test:"
        )
        
        messages = history.messages
        assert messages == []
        mock_client.get.assert_called_with("chat_test:test_session")

    def test_clear_messages(self) -> None:
        """Test clearing messages from Redis."""
        mock_client = Mock()
        mock_client.delete.return_value = 1
        
        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_client,
            key_prefix="chat_test:"
        )
        
        history.clear()
        mock_client.delete.assert_called_with("chat_test:test_session")

    def test_add_messages_with_ttl(self) -> None:
        """Test adding messages with TTL."""
        mock_client = Mock()
        mock_client.get.return_value = None  # No existing messages
        mock_client.setex.return_value = None
        
        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_client,
            key_prefix="chat_test:",
            ttl=3600
        )
        
        history.add_user_message("Hello, AI assistant!")
        
        # Verify setex was called with TTL
        assert mock_client.setex.called
        call_args = mock_client.setex.call_args
        assert call_args[0][0] == "chat_test:test_session"  # key
        assert call_args[0][1] == 3600  # TTL
        # The third argument is the JSON data, which we verify contains the message
        stored_data = json.loads(call_args[0][2])
        assert len(stored_data) == 1
        assert stored_data[0]["type"] == "human"
        assert stored_data[0]["data"]["content"] == "Hello, AI assistant!"

    def test_key_property(self) -> None:
        """Test the key property returns correct Redis key."""
        mock_client = Mock()
        
        # Test without key_prefix
        history1 = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_client
        )
        assert history1.key == "test_session"
        
        # Test with key_prefix
        history2 = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_client,
            key_prefix="chat_app:"
        )
        assert history2.key == "chat_app:test_session"

    def test_redis_import_error(self) -> None:
        """Test that ImportError is raised when redis package is not available."""
        with patch("langchain_redis.chat_message_histories.redis", None):
            with pytest.raises(ImportError, match="redis package is required"):
                RedisChatMessageHistory(
                    session_id="test_session",
                    redis_url="redis://localhost:6379"
                )