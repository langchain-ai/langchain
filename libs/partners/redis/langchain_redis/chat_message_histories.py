"""Redis-based chat message history implementation.

This module provides RedisChatMessageHistory which allows storing
chat message history in a Redis instance.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

if TYPE_CHECKING:
    import redis

logger = logging.getLogger(__name__)


class RedisChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in Redis.
    
    This class stores chat message history in a Redis instance.
    It supports session-based storage with optional key prefixes.
    
    Args:
        session_id: The session ID to use for this chat history.
        redis_url: The URL of the Redis instance to connect to.
        key_prefix: Optional prefix to add to the Redis key. Defaults to empty string.
        redis_client: Optional Redis client instance. If not provided, one will be created.
        ttl: Optional TTL in seconds for the Redis key. If None, no TTL is set.
    
    Example:
        Basic usage:
        
        .. code-block:: python
        
            from langchain_redis import RedisChatMessageHistory
            
            history = RedisChatMessageHistory(
                session_id="session_123", 
                redis_url="redis://localhost:6379"
            )
            
            history.add_user_message("Hello, AI!")
            history.add_ai_message("Hello! How can I help you?")
            
            # Get all messages
            messages = history.messages
        
        With key prefix:
        
        .. code-block:: python
        
            history = RedisChatMessageHistory(
                session_id="session_123", 
                redis_url="redis://localhost:6379",
                key_prefix="chat_app:"
            )
            # Messages will be stored under key: "chat_app:session_123"
    """

    def __init__(
        self,
        session_id: str,
        redis_url: Optional[str] = None,
        key_prefix: str = "",
        redis_client: Optional[redis.Redis] = None,
        ttl: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RedisChatMessageHistory.
        
        Args:
            session_id: The session ID to use for this chat history.
            redis_url: The URL of the Redis instance to connect to.
            key_prefix: Optional prefix to add to the Redis key.
            redis_client: Optional Redis client instance.
            ttl: Optional TTL in seconds for the Redis key.
            **kwargs: Additional arguments passed to Redis client.
        """
        if redis_client:
            self.redis_client = redis_client
        elif redis_url:
            try:
                import redis
            except ImportError as e:
                raise ImportError(
                    "redis package is required for RedisChatMessageHistory. "
                    "Please install it with `pip install redis`."
                ) from e
            
            self.redis_client = redis.from_url(redis_url, **kwargs)
        else:
            raise ValueError("Either redis_url or redis_client must be provided.")
        
        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        
        # Construct the full Redis key
        self._key = f"{key_prefix}{session_id}" if key_prefix else session_id

    @property
    def key(self) -> str:
        """Redis key for storing messages."""
        return self._key

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """Retrieve all messages from Redis.
        
        Returns:
            List of BaseMessage objects stored in Redis.
        """
        try:
            # Get the stored messages from Redis
            stored_messages = self.redis_client.get(self._key)
            
            if stored_messages:
                # Decode the JSON string and convert back to messages
                messages_data = json.loads(stored_messages.decode("utf-8"))
                return messages_from_dict(messages_data)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving messages from Redis: {e}")
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the Redis store.
        
        Args:
            messages: A sequence of BaseMessage objects to store.
        """
        if not messages:
            return
        
        try:
            # Get existing messages
            existing_messages = self.messages
            
            # Combine existing and new messages
            all_messages = existing_messages + list(messages)
            
            # Convert to serializable format
            messages_data = [message_to_dict(message) for message in all_messages]
            
            # Store in Redis
            messages_json = json.dumps(messages_data)
            if self.ttl:
                self.redis_client.setex(self._key, self.ttl, messages_json)
            else:
                self.redis_client.set(self._key, messages_json)
                
        except Exception as e:
            logger.error(f"Error storing messages to Redis: {e}")
            raise

    def clear(self) -> None:
        """Clear all messages from the Redis store."""
        try:
            self.redis_client.delete(self._key)
        except Exception as e:
            logger.error(f"Error clearing messages from Redis: {e}")
            raise

    def __del__(self) -> None:
        """Close Redis connection when the object is destroyed."""
        try:
            if hasattr(self, 'redis_client') and hasattr(self.redis_client, 'close'):
                self.redis_client.close()
        except Exception:
            # Ignore errors during cleanup
            pass