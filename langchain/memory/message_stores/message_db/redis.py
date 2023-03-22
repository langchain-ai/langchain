import logging
import pickle
import redis
from time import time
from typing import List, Optional


from langchain.schema import BaseMessage,  MessageDB, messages_from_dict, messages_to_dict, _message_to_dict


logger = logging.getLogger(__name__)


class RedisMessageDB(MessageDB):

    def __init__(self,
                 url: str = 'redis://localhost:6379/0',
                 key_prefix: str = 'chatsession:',
                 ttl: Optional[int] = None):
        try:
            self.redis = redis.Redis.from_url(url=url)
        except redis.exceptions.ConnectionError as error:
            logger.error(error)

        self.key_prefix = key_prefix
        self.ttl = ttl

    def get_key(self, session_id: str) -> str:
        """Construct the record key to use"""
        return self.key_prefix + session_id

    def read(self, session_id: str) -> List[BaseMessage]:
        """Retrieve the messages from Redis"""
        if self.redis.exists(self.get_key(session_id)):
            items = pickle.loads(self.redis.get(self.get_key(session_id)))
        else:
            items = []

        messages = messages_from_dict(items)
        return messages

    def append(self, session_id, message: BaseMessage) -> None:
        """Append the message to the record in Redis"""
        messages = self.read(session_id)
        _message = _message_to_dict(message)
        messages.append(_message)

        _exat = int(time()) + self.ttl if self.ttl else None
        self.redis.set(name=self.get_key(session_id), value=pickle.dumps(messages), exat=_exat)

    def clear(self, session_id: str) -> None:
        """Clear session memory from Redis"""
        self.redis.delete(self.get_key(session_id))
