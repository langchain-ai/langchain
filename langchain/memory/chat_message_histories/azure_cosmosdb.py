import logging
from typing import List

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
)

logger = logging.getLogger(__name__)


class AzureCosmosDBChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        endpoint: str,
        credential: str,
        database_name: str = "chat_history",
        container_name: str = "messages",
        ttl: str = -1,
        connection_verify: bool = True,
    ):
        try:
            from azure.cosmos import CosmosClient
        except ImportError:
            raise ValueError(
                "Could not import azure-cosmos python package. "
                "Please install it with `pip install azure-cosmos`."
            )

        try:
            self.cosmos_client = CosmosClient(
                endpoint, credential, connection_verify=connection_verify
            )
        except Exception as error:
            logger.error(error)

        self.session_id = session_id
        self.database_name = database_name
        self.container_name = container_name
        self.ttl = ttl

        self.database = self._get_database()
        self.container = self._get_container()

    def _get_database(self):
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return self.cosmos_client.get_database_client(self.database_name)
        except ResourceNotFoundError:
            return self.cosmos_client.create_database(self.database_name)

    def _get_container(self):
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return self.database.get_container_client(self.container_name)
        except ResourceNotFoundError:
            return self.database.create_container(
                self.container_name, partition_key="/id"
            )

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Cosmos DB."""
        from azure.cosmos import errors

        try:
            items = self.container.read_item(
                item=self.session_id, partition_key=self.session_id
            )
            messages = messages_from_dict(items["messages"])
            return messages
        except errors.CosmosHttpResponseError as error:
            if error.status_code == 404:  # Not Found
                return []
            else:
                raise error

    def add_user_message(self, message: str) -> None:
        self.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.append(AIMessage(content=message))

    def append(self, message: BaseMessage) -> None:
        """Append the message to the record in Cosmos DB."""
        messages = self.messages
        messages.append(message)
        item = {
            "id": self.session_id,
            "messages": [_message_to_dict(m) for m in messages],
            "ttl": self.ttl,
        }
        self.container.upsert_item(item)

    def clear(self) -> None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            self.container.delete_item(
                item=self.session_id, partition_key=self.session_id
            )
        except ResourceNotFoundError:
            pass
