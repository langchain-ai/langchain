import json
import logging
from typing import List, Optional

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)


class RedisChatMessageHistory(BaseChatMessageHistory):
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
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        try:
            self.redis_client = redis.Redis.from_url(url=url)
        except redis.exceptions.ConnectionError as error:
            logger.error(error)

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl

    def get_key(self, session_id: str) -> str:
        """Construct the record key to use"""
        return self.key_prefix + session_id

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Redis"""
        if self.redis_client.exists(self.get_key(self.session_id)):
            _r = self.redis_client.get(self.get_key(self.session_id))
            items = json.loads(_r.decode("utf-8")) if _r else []
        else:
            items = []

        messages = messages_from_dict(items)
        return messages

    def add_user_message(self, message: str) -> None:
        self.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.append(AIMessage(content=message))

    def append(self, message: BaseMessage) -> None:
        """Append the message to the record in Redis"""
        messages = messages_to_dict(self.messages)
        _message = _message_to_dict(message)
        messages.append(_message)

        self.redis_client.set(
            name=self.get_key(self.session_id), value=json.dumps(messages), ex=self.ttl
        )

    def clear(self) -> None:
        """Clear session memory from Redis"""
        self.redis_client.delete(self.get_key(self.session_id))
