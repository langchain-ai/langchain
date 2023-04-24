"""Azure CosmosDB Memory History."""
from __future__ import annotations

import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, List, Optional, Type

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from azure.cosmos import ContainerProxy, CosmosClient


class CosmosDBChatMessageHistory(BaseChatMessageHistory):
    """Chat history backed by Azure CosmosDB."""

    def __init__(
        self,
        cosmos_endpoint: str,
        cosmos_database: str,
        cosmos_container: str,
        credential: Any,
        session_id: str,
        user_id: str,
        ttl: Optional[int] = None,
    ):
        """
        Initializes a new instance of the CosmosDBChatMessageHistory class.

        :param cosmos_endpoint: The connection endpoint for the Azure Cosmos DB account.
        :param cosmos_database: The name of the database to use.
        :param cosmos_container: The name of the container to use.
        :param credential: The credential to use to authenticate to Azure Cosmos DB.
        :param session_id: The session ID to use, can be overwritten while loading.
        :param user_id: The user ID to use, can be overwritten while loading.
        :param ttl: The time to live (in seconds) to use for documents in the container.
        """
        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_database = cosmos_database
        self.cosmos_container = cosmos_container
        self.credential = credential
        self.session_id = session_id
        self.user_id = user_id
        self.ttl = ttl

        self._client: Optional[CosmosClient] = None
        self._container: Optional[ContainerProxy] = None
        self.messages: List[BaseMessage] = []

    def prepare_cosmos(self) -> None:
        """Prepare the CosmosDB client.

        Use this function or the context manager to make sure your database is ready.
        """
        try:
            from azure.cosmos import (  # pylint: disable=import-outside-toplevel # noqa: E501
                CosmosClient,
                PartitionKey,
            )
        except ImportError as exc:
            raise ImportError(
                "You must install the azure-cosmos package to use the CosmosDBChatMessageHistory."  # noqa: E501
            ) from exc
        self._client = CosmosClient(
            url=self.cosmos_endpoint, credential=self.credential
        )
        database = self._client.create_database_if_not_exists(self.cosmos_database)
        self._container = database.create_container_if_not_exists(
            self.cosmos_container,
            partition_key=PartitionKey("/user_id"),
            default_ttl=self.ttl,
        )
        self.load_messages()

    def __enter__(self) -> "CosmosDBChatMessageHistory":
        """Context manager entry point."""
        if self._client:
            self._client.__enter__()
            self.prepare_cosmos()
            return self
        raise ValueError("Client not initialized")

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Context manager exit"""
        self.upsert_messages()
        if self._client:
            self._client.__exit__(exc_type, exc_val, traceback)

    def load_messages(self) -> None:
        """Retrieve the messages from Cosmos"""
        if not self._container:
            raise ValueError("Container not initialized")
        try:
            from azure.cosmos.exceptions import (  # pylint: disable=import-outside-toplevel # noqa: E501
                CosmosHttpResponseError,
            )
        except ImportError as exc:
            raise ImportError(
                "You must install the azure-cosmos package to use the CosmosDBChatMessageHistory."  # noqa: E501
            ) from exc
        try:
            item = self._container.read_item(
                item=self.session_id, partition_key=self.user_id
            )
        except CosmosHttpResponseError:
            logger.info("no session found")
            return
        if (
            "messages" in item
            and len(item["messages"]) > 0
            and isinstance(item["messages"][0], list)
        ):
            self.messages = messages_from_dict(item["messages"])

    def add_user_message(self, message: str) -> None:
        """Add a user message to the memory."""
        self.upsert_messages(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Add a AI message to the memory."""
        self.upsert_messages(AIMessage(content=message))

    def upsert_messages(self, new_message: Optional[BaseMessage] = None) -> None:
        """Update the cosmosdb item."""
        if new_message:
            self.messages.append(new_message)
        if not self._container:
            raise ValueError("Container not initialized")
        self._container.upsert_item(
            body={
                "id": self.session_id,
                "user_id": self.user_id,
                "messages": messages_to_dict(self.messages),
            }
        )

    def clear(self) -> None:
        """Clear session memory from this memory and cosmos."""
        self.messages = []
        if self._container:
            self._container.delete_item(
                item=self.session_id, partition_key=self.user_id
            )
