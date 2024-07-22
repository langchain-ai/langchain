import json
import logging
from typing import Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from pymongo import MongoClient, errors

logger = logging.getLogger(__name__)

DEFAULT_DBNAME = "chat_history"
DEFAULT_COLLECTION_NAME = "message_store"
DEFAULT_SESSION_ID_KEY = "SessionId"
DEFAULT_HISTORY_KEY = "History"


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in MongoDB.

    Args:
        connection_string: connection string to connect to MongoDB
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
        database_name: name of the database to use
        collection_name: name of the collection to use
        session_id_key: name of the field that stores the session id
        history_key: name of the field that stores the chat history
        create_index: whether to create an index on the session id field
        index_kwargs: additional keyword arguments to pass to the index creation
    """

    def __init__(
        self,
        connection_string: str,
        session_id: str,
        database_name: str = DEFAULT_DBNAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        session_id_key: str = DEFAULT_SESSION_ID_KEY,
        history_key: str = DEFAULT_HISTORY_KEY,
        create_index: bool = True,
        index_kwargs: Optional[Dict] = None,
    ):
        self.connection_string = connection_string
        self.session_id = session_id
        self.database_name = database_name
        self.collection_name = collection_name
        self.session_id_key = session_id_key
        self.history_key = history_key

        try:
            self.client: MongoClient = MongoClient(connection_string)
        except errors.ConnectionFailure as error:
            logger.error(error)

        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

        if create_index:
            index_kwargs = index_kwargs or {}
            self.collection.create_index(self.session_id_key, **index_kwargs)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from MongoDB"""
        try:
            cursor = self.collection.find({self.session_id_key: self.session_id})
        except errors.OperationFailure as error:
            logger.error(error)

        if cursor:
            items = [json.loads(document[self.history_key]) for document in cursor]
        else:
            items = []

        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in MongoDB"""
        try:
            self.collection.insert_one(
                {
                    self.session_id_key: self.session_id,
                    self.history_key: json.dumps(message_to_dict(message)),
                }
            )
        except errors.WriteError as err:
            logger.error(err)

    def clear(self) -> None:
        """Clear session memory from MongoDB"""
        try:
            self.collection.delete_many({self.session_id_key: self.session_id})
        except errors.WriteError as err:
            logger.error(err)
