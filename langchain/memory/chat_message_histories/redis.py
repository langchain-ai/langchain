import json
import logging
from typing import Any, List, Optional

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
)

try:
    import redis
except ImportError:
    raise ValueError(
        "Could not import redis python package. "
        "Please install it with `pip install redis`."
    )

logger = logging.getLogger(__name__)


class RedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        key_prefix: str = "message_store:",
        ttl: Optional[int] = None,
        redis_client: Optional[redis.client] = None,
    ):

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.redis_client = redis_client

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

    def from_url(self, url: str, **kwargs: Any) -> None:
        """Constructs redis client from url"""
        try:
            self.redis_client = redis.Redis.from_url(url=url, **kwargs)
        except redis.exceptions.ConnectionError as error:
            logger.error(error)

    def add_user_message(self, message: str) -> None:
        self.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.append(AIMessage(content=message))

    def append(self, message: BaseMessage) -> None:
        """Append the message to the record in Redis"""
        self.redis_client.lpush(self.key, json.dumps(_message_to_dict(message)))
        if self.ttl:
            self.redis_client.expire(self.key, self.ttl)

    def clear(self) -> None:
        """Clear session memory from Redis"""
        self.redis_client.delete(self.key)

    def ping(self) -> bool:
        """Pings Redis DB"""
        return self.redis_client.ping()
