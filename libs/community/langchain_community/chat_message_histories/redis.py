import json
import logging
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

from langchain_community.utilities.redis import get_client

logger = logging.getLogger(__name__)


class RedisChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Redis database.

    Setup:
        Install ``redis`` python package.

        .. code-block:: bash

            pip install redis

    Instantiate:
        .. code-block:: python

        from langchain_community.chat_message_histories import RedisChatMessageHistory

        history = RedisChatMessageHistory(
            session_id = "your-session-id",
            url="redis://your-host:your-port:your-database",  # redis://localhost:6379/0
        )

    Add and retrieve messages:
        .. code-block:: python

            # Add single message
            history.add_message(message)

            # Add batch messages
            history.add_messages([message1, message2, message3, ...])

            # Add human message
            history.add_user_message(human_message)

            # Add ai message
            history.add_ai_message(ai_message)

            # Retrieve messages
            messages = history.messages
    """  # noqa: E501

    def __init__(
        self,
        session_id: str,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "message_store:",
        ttl: Optional[int] = None,
    ):
        """Initialize with a RedisChatMessageHistory instance.

        Args:
            session_id: str
                The ID for single chat session. Used to form keys with `key_prefix`.
            url: Optional[str]
                String parameter configuration for connecting to the redis.
            key_prefix: Optional[str]
                The prefix of the key, combined with `session id` to form the key.
            ttl: Optional[int]
                Set the expiration time of `key`, the unit is seconds.
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        try:
            self.redis_client = get_client(redis_url=url)
        except redis.exceptions.ConnectionError as error:
            logger.error(error)

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl

    @property
    def key(self) -> str:
        """Construct the record key to use"""
        return self.key_prefix + self.session_id

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from Redis"""
        _items = self.redis_client.lrange(self.key, 0, -1)
        items = [json.loads(m.decode("utf-8")) for m in _items[::-1]]
        messages = messages_from_dict(items)
        return messages

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError(
            "Direct assignment to 'messages' is not allowed."
            " Use the 'add_messages' instead."
        )

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Redis"""
        self.redis_client.lpush(self.key, json.dumps(message_to_dict(message)))
        if self.ttl:
            self.redis_client.expire(self.key, self.ttl)

    def clear(self) -> None:
        """Clear session memory from Redis"""
        self.redis_client.delete(self.key)
