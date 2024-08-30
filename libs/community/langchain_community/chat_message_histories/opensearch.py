import json
import logging
from time import time
from typing import TYPE_CHECKING, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

if TYPE_CHECKING:
    from opensearchpy import OpenSearch

logger = logging.getLogger(__name__)

DEFAULT_INDEX_NAME = "chat-history"
IMPORT_OPENSEARCH_PY_ERROR = (
    "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
)


def _default_message_mapping() -> dict:
    return {"mappings": {"properties": {"SessionId": {"type": "keyword"}}}}


class OpenSearchChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in OpenSearch.

    Args:
        opensearch_url: connection string to connect to OpenSearch.
        session_id: Arbitrary key that is used to store the messages
            of a single chat session.
        index: Name of the index to use.
    """

    def __init__(
        self,
        opensearch_url: str = None,
        session_id: str = None,
        index: str = DEFAULT_INDEX_NAME,
        **kwargs,
    ):
        self.opensearch_url = opensearch_url
        self.index = index
        self.session_id = session_id

        opensearch_connection = kwargs.get("opensearch_connection")

        try:
            from opensearchpy import OpenSearch
        except ImportError as e:
            raise ImportError(IMPORT_OPENSEARCH_PY_ERROR) from e

        # Initialize OpenSearch client from passed client arg or connection info
        if not opensearch_url and not opensearch_connection:
            raise ValueError("OpenSearch connection or URL is required.")

        try:
            if opensearch_connection:
                self.client = kwargs.get("opensearch_connection").options(
                    headers={"user-agent": self.get_user_agent()}
                )
            else:
                self.client = OpenSearch(
                    [opensearch_url],
                    **kwargs,
                )
        except ValueError as e:
            raise ValueError(
                "Your OpenSearch client string is mis-formatted. Got error: {e}"
            ) from e

        if self.client.indices.exists(index=index):
            logger.debug(
                "Chat history index %s already exists, skipping creation.", index
            )
        else:
            logger.debug("Creating index %s for storing chat history.", index)

            self.client.indices.create(
                index=index,
                body=_default_message_mapping(),
            )

    @staticmethod
    def get_user_agent() -> str:
        from langchain_community import __version__

        return f"langchain-py-ms/{__version__}"

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from OpenSearch"""
        try:
            from opensearchpy import RequestError

            result = self.client.search(
                index=self.index,
                body={"query": {"match": {"SessionId": self.session_id}}},
            )
        except RequestError as err:
            logger.error("Could not retrieve messages from OpenSearch: %s", err)
            raise err

        if result and len(result["hits"]["hits"]) > 0:
            items = [
                json.loads(document["_source"]["History"])
                for document in result["hits"]["hits"]
            ]
        else:
            items = []

        return messages_from_dict(items)

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError(
            "Direct assignment to 'messages' is not allowed."
            " Use the 'add_messages' instead."
        )

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat session in OpenSearch"""
        try:
            from opensearchpy import RequestError

            self.client.index(
                index=self.index,
                body={
                    "SessionId": self.session_id,
                    "Created_At": round(time() * 1000),
                    "History": json.dumps(message_to_dict(message)),
                },
                refresh=True,
            )
        except RequestError as err:
            logger.error("Could not add message to OpenSearch: %s", err)
            raise err

    def clear(self) -> None:
        """Clear session memory in OpenSearch"""
        try:
            from opensearchpy import RequestError

            self.client.delete_by_query(
                index=self.index,
                body={"SessionId": self.session_id},
            )
        except RequestError as err:
            logger.error("Could not clear session memory in OpenSearch: %s", err)
            raise err
