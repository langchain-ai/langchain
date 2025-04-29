import json
import logging
from typing import List

from langchain_core._api.deprecation import deprecated
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

logger = logging.getLogger(__name__)

DEFAULT_DBNAME = "chat_history"
DEFAULT_COLLECTION_NAME = "message_store"


@deprecated(
    since="0.0.25",
    removal="1.0",
    alternative_import="langchain_mongodb.MongoDBChatMessageHistory",
)
class MongoDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in MongoDB.

    Args:
        connection_string: connection string to connect to MongoDB
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
        database_name: name of the database to use
        collection_name: name of the collection to use
        create_index: whether to create an index with name SessionId. Set to False if
            such an index already exists.
    """

    def __init__(
        self,
        connection_string: str,
        session_id: str,
        database_name: str = DEFAULT_DBNAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        create_index: bool = True,
    ):
        from pymongo import MongoClient, errors

        self.connection_string = connection_string
        self.session_id = session_id
        self.database_name = database_name
        self.collection_name = collection_name

        try:
            self.client: MongoClient = MongoClient(connection_string)
        except errors.ConnectionFailure as error:
            logger.error(error)

        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        if create_index:
            self.collection.create_index("SessionId")

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        """Retrieve the messages from MongoDB"""
        from pymongo import errors

        try:
            cursor = self.collection.find({"SessionId": self.session_id})
        except errors.OperationFailure as error:
            logger.error(error)

        if cursor:
            items = [json.loads(document["History"]) for document in cursor]
        else:
            items = []

        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in MongoDB"""
        from pymongo import errors

        try:
            self.collection.insert_one(
                {
                    "SessionId": self.session_id,
                    "History": json.dumps(message_to_dict(message)),
                }
            )
        except errors.WriteError as err:
            logger.error(err)

    def clear(self) -> None:
        """Clear session memory from MongoDB"""
        from pymongo import errors

        try:
            self.collection.delete_many({"SessionId": self.session_id})
        except errors.WriteError as err:
            logger.error(err)
