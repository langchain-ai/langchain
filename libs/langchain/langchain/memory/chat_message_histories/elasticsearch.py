import json
import logging
from time import time
from typing import TYPE_CHECKING, List

from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in Elasticsearch.

    Args:
        client: Elasticsearch client.
        index: name of the index to use.
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
    """

    def __init__(
        self,
        client: Elasticsearch,
        index: str,
        session_id: str,
    ):
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        self.client: Elasticsearch = client
        self.index: str = index
        self.session_id: str = session_id

        if client.indices.exists(index=index):
            logger.debug(
                f"Chat history index {index} already exists, skipping creation."
            )
        else:
            logger.debug(f"Creating index {index} for storing chat history.")

            client.indices.create(
                index=index,
                mappings={
                    "properties": {
                        "session_id": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "history": {"type": "text"},
                    }
                },
            )

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        """Retrieve the messages from Elasticsearch"""
        try:
            from elasticsearch import ApiError

            result = self.client.search(
                index=self.index,
                query={"term": {"session_id": self.session_id}},
                sort="created_at:asc",
            )
        except ApiError as err:
            logger.error(err)

        if result and len(result["hits"]["hits"]) > 0:
            items = [
                json.loads(document["_source"]["history"])
                for document in result["hits"]["hits"]
            ]
        else:
            items = []

        return messages_from_dict(items)

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat session in Elasticsearch"""
        try:
            from elasticsearch import ApiError

            print("indexing message", message)
            self.client.index(
                index=self.index,
                document={
                    "session_id": self.session_id,
                    "created_at": round(time() * 1000),
                    "history": json.dumps(_message_to_dict(message)),
                },
                refresh=True,
            )
        except ApiError as err:
            logger.error(err)

    def clear(self) -> None:
        """Clear session memory in Elasticsearch"""
        try:
            from elasticsearch import ApiError

            self.client.delete_by_query(
                index=self.index,
                query={"term": {"session_id": self.session_id}},
                refresh=True,
            )
        except ApiError:
            logger.error("Could not clear session memory in Elasticsearch")
