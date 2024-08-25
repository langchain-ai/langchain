import json
import logging
from typing import TYPE_CHECKING, List, Optional, Sequence, TypeVar, Union

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

from langchain_community.utilities.redis import get_client

if TYPE_CHECKING:
    from redis.asyncio.client import Pipeline as AsyncPipeline
    from redis.asyncio.client import Redis as AsyncRedis
    from redis.client import Pipeline, Redis

TPipeline = TypeVar("TPipeline", "Pipeline", "AsyncPipeline")

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
        *,
        redis_client: Optional["Redis"] = None,
        async_redis_client: Optional["AsyncRedis"] = None,
        pipeline_transaction: bool = True,
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

        self._redis_client = redis_client
        self._async_redis_client = async_redis_client
        self.pipeline_transaction = pipeline_transaction

        if redis_client is None and async_redis_client is None:
            try:
                self._redis_client = get_client(redis_url=url)
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
        pipeline = self.redis_client.pipeline(transaction=False)
        pipeline_results = self._construct_get_messages_pipeline(pipeline).execute()
        return self._parse_messages(pipeline_results[0])

    async def aget_messages(self) -> List[BaseMessage]:
        pipeline = self.async_redis_client.pipeline(transaction=False)
        pipeline_results = await self._construct_get_messages_pipeline(
            pipeline
        ).execute()
        return self._parse_messages(pipeline_results[0])

    def _construct_get_messages_pipeline(self, pipeline: TPipeline) -> TPipeline:
        return pipeline.lrange(self.key, 0, -1)

    @staticmethod
    def _parse_messages(items: list[Union[str, bytes]]) -> List[BaseMessage]:
        json_items = [json.loads(m.decode("utf-8")) for m in items[::-1]]
        messages = messages_from_dict(json_items)
        return messages

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError(
            "Direct assignment to 'messages' is not allowed."
            " Use the 'add_messages' instead."
        )

    def clear(self) -> None:
        """Clear session memory from Redis"""
        self.redis_client.delete(self.key)

    async def aclear(self) -> None:
        await self.async_redis_client.delete(self.key)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        pipeline = self.redis_client.pipeline(transaction=self.pipeline_transaction)
        self._construct_add_messages_to_pipeline(messages, pipeline).execute()

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        pipeline = self.async_redis_client.pipeline(
            transaction=self.pipeline_transaction
        )
        await self._construct_add_messages_to_pipeline(messages, pipeline).execute()

    def _construct_add_messages_to_pipeline(
        self, messages: Sequence[BaseMessage], pipeline: TPipeline
    ) -> TPipeline:
        for message in messages:
            pipeline.lpush(self.key, self._encode_message(message))

        if self.ttl:
            pipeline.expire(self.key, self.ttl)

        return pipeline

    @staticmethod
    def _encode_message(message: BaseMessage) -> str:
        return json.dumps(message_to_dict(message))

    @property
    def redis_client(self) -> "Redis":
        if self._redis_client is None:
            raise RuntimeError("Sync redis client is not initialized.")

        return self._redis_client

    @property
    def async_redis_client(self) -> "AsyncRedis":
        if self._async_redis_client is None:
            raise RuntimeError("Async redis client is not initialized.")

        return self._async_redis_client
