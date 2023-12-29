import json
import logging
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

from langchain_community.utilities.redis import get_client

logger = logging.getLogger(__name__)


class RedisChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Redis database."""

    def __init__(
        self,
        session_id: str,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "message_store:",
        ttl: Optional[int] = None,
    ):
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
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Redis"""
        _items = self.redis_client.lrange(self.key, 0, -1)
        items = [json.loads(m.decode("utf-8")) for m in _items[::-1]]
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Redis"""
        self.redis_client.lpush(self.key, json.dumps(message_to_dict(message)))
        if self.ttl:
            self.redis_client.expire(self.key, self.ttl)

    def clear(self) -> None:
        """Clear session memory from Redis"""
        self.redis_client.delete(self.key)


class RedisChatMessageHistoryWithTokenLimit(RedisChatMessageHistory):
    """Chat message history stored in a Redis database, with a max token limit"""

    def __init__(
        self,
        session_id: str,
        llm: BaseLanguageModel,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "message_store:",
        ttl: Optional[int] = None,
        max_token_limit=2000,
        retain_messages=True,
    ):
        super().__init__(session_id=session_id, url=url, key_prefix=key_prefix, ttl=ttl)
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.retain_messages = retain_messages

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from Redis, pruned until below max token limit"""
        messages = super().messages
        while self.llm.get_num_tokens_from_messages(messages) > self.max_token_limit:
            del messages[0]
            if not self.retain_messages:
                self.redis_client.rpop(self.key)
        return messages
