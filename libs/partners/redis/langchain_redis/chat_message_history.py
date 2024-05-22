import json  # noqa: I001
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict
from redis import Redis
from redis.exceptions import ResponseError
from redis.commands.json.path import Path
from redis.commands.search.field import NumericField, TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query


class RedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "chat:",
        ttl: Optional[int] = None,
        index_name: str = "idx:chat_history",
        **kwargs: Any,
    ):
        self.redis_client = Redis.from_url(redis_url, **kwargs)
        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.index_name = index_name
        self._ensure_index()

    @property
    def id(self) -> str:
        return self.session_id

    def _ensure_index(self) -> None:
        try:
            self.redis_client.ft(self.index_name).info()
        except ResponseError as e:
            if str(e) == "Unknown index name":
                schema = (
                    TagField("$.session_id", as_name="session_id"),
                    TextField("$.data.content", as_name="content"),
                    TagField("$.type", as_name="type"),
                    NumericField("$.timestamp", as_name="timestamp"),
                )
                definition = IndexDefinition(
                    prefix=[self.key_prefix], index_type=IndexType.JSON
                )
                self.redis_client.ft(self.index_name).create_index(
                    schema, definition=definition
                )
            else:
                raise

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        query = (
            Query(f"@session_id:{{{self.session_id}}}")
            .sort_by("timestamp", asc=True)
            .paging(0, 10000)
        )
        results = self.redis_client.ft(self.index_name).search(query)
        return messages_from_dict(
            [
                {
                    "type": json.loads(doc.json)["type"],
                    "data": json.loads(doc.json)["data"],
                }
                for doc in results.docs
            ]
        )

    def add_message(self, message: BaseMessage) -> None:
        data_to_store = {
            "type": message.type,
            "data": {
                "content": message.content,
                "additional_kwargs": message.additional_kwargs,
                "type": message.type,
            },
            "session_id": self.session_id,
            "timestamp": datetime.now().timestamp(),
        }

        key = f"{self.key_prefix}{self.session_id}:{data_to_store['timestamp']}"
        self.redis_client.json().set(key, Path.root_path(), data_to_store)

        if self.ttl:
            self.redis_client.expire(key, self.ttl)

    def clear(self) -> None:
        query = Query(f"@session_id:{{{self.session_id}}}").paging(0, 10000)
        results = self.redis_client.ft(self.index_name).search(query)
        for doc in results.docs:
            self.redis_client.delete(doc.id)

    def search_messages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        search_query = (
            Query(f"(@session_id:{{{self.session_id}}}) (@content:{query})")
            .sort_by("timestamp", asc=True)
            .paging(0, limit)
        )
        results = self.redis_client.ft(self.index_name).search(search_query)

        return [json.loads(doc.json)["data"] for doc in results.docs]

    def __len__(self) -> int:
        query = Query(f"@session_id:{{{self.session_id}}}")
        return self.redis_client.ft(self.index_name).search(query).total
