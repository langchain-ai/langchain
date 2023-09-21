import json
import logging
from typing import List, Optional

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict

logger = logging.getLogger(__name__)


class UpstashRedisChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in an Upstash Redis database."""

    def __init__(
        self,
        session_id: str,
        url: str = "",
        token: str = "",
        key_prefix: str = "message_store:",
        ttl: Optional[int] = None,
    ):
        try:
            from upstash_redis import Redis
        except ImportError:
            raise ImportError(
                "Could not import upstash redis python package. "
                "Please install it with `pip install upstash_redis`."
            )

        if url == "" or token == "":
            raise ValueError(
                "UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN are needed."
            )

        try:
            self.redis_client = Redis(url=url, token=token)
        except Exception:
            logger.error("Upstash Redis instance could not be initiated.")

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl

    @property
    def key(self) -> str:
        """Construct the record key to use"""
        return self.key_prefix + self.session_id

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Upstash Redis"""
        _items = self.redis_client.lrange(self.key, 0, -1)
        items = [json.loads(m) for m in _items[::-1]]
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Upstash Redis"""
        self.redis_client.lpush(self.key, json.dumps(_message_to_dict(message)))
        if self.ttl:
            self.redis_client.expire(self.key, self.ttl)

    def clear(self) -> None:
        """Clear session memory from Upstash Redis"""
        self.redis_client.delete(self.key)
