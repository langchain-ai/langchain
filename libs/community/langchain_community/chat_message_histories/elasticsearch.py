import json
import logging
from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core._api import deprecated
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


@deprecated("0.0.27", alternative="Use langchain-elasticsearch package", pending=True)
class ElasticsearchChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in Elasticsearch.

    Args:
        es_url: URL of the Elasticsearch instance to connect to.
        es_cloud_id: Cloud ID of the Elasticsearch instance to connect to.
        es_user: Username to use when connecting to Elasticsearch.
        es_password: Password to use when connecting to Elasticsearch.
        es_api_key: API key to use when connecting to Elasticsearch.
        es_connection: Optional pre-existing Elasticsearch connection.
        esnsure_ascii: Used to escape ASCII symbols in json.dumps. Defaults to True.
        index: Name of the index to use.
        session_id: Arbitrary key that is used to store the messages
            of a single chat session.
    """

    def __init__(
        self,
        index: str,
        session_id: str,
        *,
        es_connection: Optional["Elasticsearch"] = None,
        es_url: Optional[str] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_api_key: Optional[str] = None,
        es_password: Optional[str] = None,
        esnsure_ascii: Optional[bool] = True,
    ):
        self.index: str = index
        self.session_id: str = session_id
        self.ensure_ascii = esnsure_ascii

        # Initialize Elasticsearch client from passed client arg or connection info
        if es_connection is not None:
            self.client = es_connection.options(
                headers={"user-agent": self.get_user_agent()}
            )
        elif es_url is not None or es_cloud_id is not None:
            self.client = ElasticsearchChatMessageHistory.connect_to_elasticsearch(
                es_url=es_url,
                username=es_user,
                password=es_password,
                cloud_id=es_cloud_id,
                api_key=es_api_key,
            )
        else:
            raise ValueError(
                """Either provide a pre-existing Elasticsearch connection, \
                or valid credentials for creating a new connection."""
            )

        if self.client.indices.exists(index=index):
            logger.debug(
                f"Chat history index {index} already exists, skipping creation."
            )
        else:
            logger.debug(f"Creating index {index} for storing chat history.")

            self.client.indices.create(
                index=index,
                mappings={
                    "properties": {
                        "session_id": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "history": {"type": "text"},
                    }
                },
            )

    @staticmethod
    def get_user_agent() -> str:
        from langchain_community import __version__

        return f"langchain-py-ms/{__version__}"

    @staticmethod
    def connect_to_elasticsearch(
        *,
        es_url: Optional[str] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> "Elasticsearch":
        try:
            import elasticsearch
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        if es_url and cloud_id:
            raise ValueError(
                "Both es_url and cloud_id are defined. Please provide only one."
            )

        connection_params: Dict[str, Any] = {}

        if es_url:
            connection_params["hosts"] = [es_url]
        elif cloud_id:
            connection_params["cloud_id"] = cloud_id
        else:
            raise ValueError("Please provide either elasticsearch_url or cloud_id.")

        if api_key:
            connection_params["api_key"] = api_key
        elif username and password:
            connection_params["basic_auth"] = (username, password)

        es_client = elasticsearch.Elasticsearch(
            **connection_params,
            headers={"user-agent": ElasticsearchChatMessageHistory.get_user_agent()},
        )
        try:
            es_client.info()
        except Exception as err:
            logger.error(f"Error connecting to Elasticsearch: {err}")
            raise err

        return es_client

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
            logger.error(f"Could not retrieve messages from Elasticsearch: {err}")
            raise err

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

            self.client.index(
                index=self.index,
                document={
                    "session_id": self.session_id,
                    "created_at": round(time() * 1000),
                    "history": json.dumps(
                        message_to_dict(message),
                        ensure_ascii=bool(self.ensure_ascii),
                    ),
                },
                refresh=True,
            )
        except ApiError as err:
            logger.error(f"Could not add message to Elasticsearch: {err}")
            raise err

    def clear(self) -> None:
        """Clear session memory in Elasticsearch"""
        try:
            from elasticsearch import ApiError

            self.client.delete_by_query(
                index=self.index,
                query={"term": {"session_id": self.session_id}},
                refresh=True,
            )
        except ApiError as err:
            logger.error(f"Could not clear session memory in Elasticsearch: {err}")
            raise err
