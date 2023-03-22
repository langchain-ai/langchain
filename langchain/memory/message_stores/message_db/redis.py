import json
import logging
from typing import Any, Dict, List, Optional, Union

from langchain.schema import (
    BaseMessage,
    MessageDB,
    _message_to_dict,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)


class RedisMessageDB(MessageDB):
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "message_store:",
        ttl: Optional[int] = None,
    ):
        try:
            import redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        try:
            self.redis_client = redis.Redis.from_url(url=url)
        except redis.exceptions.ConnectionError as error:
            logger.error(error)

        self.key_prefix = key_prefix
        self.ttl = ttl

    def get_key(self, session_id: str) -> str:
        """Construct the record key to use"""
        return self.key_prefix + session_id

    def read(
        self, session_id: str, as_dict: bool = False
    ) -> Union[List[BaseMessage], List[Dict[str, Any]]]:
        """Retrieve the messages from Redis"""
        if self.redis_client.exists(self.get_key(session_id)):
            items = json.loads(
                self.redis_client.get(self.get_key(session_id)).decode("utf-8")
            )
        else:
            items = []

        if as_dict:
            return items

        messages = messages_from_dict(items)
        return messages

    def append(self, session_id: str, message: BaseMessage) -> None:
        """Append the message to the record in Redis"""
        messages = self.read(session_id, as_dict=True)
        _message = _message_to_dict(message)
        messages.append(_message)

        self.redis_client.set(
            name=self.get_key(session_id), value=json.dumps(messages), ex=self.ttl
        )

    def clear(self, session_id: str) -> None:
        """Clear session memory from Redis"""
        self.redis_client.delete(self.get_key(session_id))
