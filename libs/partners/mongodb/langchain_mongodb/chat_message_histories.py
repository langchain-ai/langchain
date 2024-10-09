import json
import logging
from typing import Dict, List, Optional, Sequence

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

try:
    from motor.motor_asyncio import AsyncIOMotorClient

    _motor_available = True
except ImportError:
    AsyncIOMotorClient = None
    _motor_available = False
    logger.warning(
        "Motor library is not installed. Asynchronous methods will fall back to using "
        "`run_in_executor`, which is less efficient. "
        "Install motor with `pip install motor` for improved performance."
    )


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in MongoDB.

    Setup:
        Install ``langchain-mongodb`` python package.

        .. code-block:: bash

            pip install langchain-mongodb

    Instantiate:
        .. code-block:: python

            from langchain_mongodb import MongoDBChatMessageHistory


            history = MongoDBChatMessageHistory(
                connection_string="mongodb://your-host:your-port/",  # mongodb://localhost:27017/
                session_id = "your-session-id",
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
        connection_string: str,
        session_id: str,
        database_name: str = DEFAULT_DBNAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        session_id_key: str = DEFAULT_SESSION_ID_KEY,
        history_key: str = DEFAULT_HISTORY_KEY,
        create_index: bool = True,
        history_size: Optional[int] = None,
        index_kwargs: Optional[Dict] = None,
    ):
        """Initialize with a MongoDBChatMessageHistory instance.

        Args:
            connection_string: str
                connection string to connect to MongoDB.
            session_id: str
                arbitrary key that is used to store the messages of
                 a single chat session.
            database_name: Optional[str]
                name of the database to use.
            collection_name: Optional[str]
                name of the collection to use.
            session_id_key: Optional[str]
                name of the field that stores the session id.
            history_key: Optional[str]
                name of the field that stores the chat history.
            create_index: Optional[bool]
                whether to create an index on the session id field.
            history_size: Optional[int]
                count of (most recent) messages to fetch from MongoDB.
            index_kwargs: Optional[Dict]
                additional keyword arguments to pass to the index creation.
        """
        self.connection_string = connection_string
        self.session_id = session_id
        self.database_name = database_name
        self.collection_name = collection_name
        self.session_id_key = session_id_key
        self.history_key = history_key
        self.history_size = history_size

        try:
            self.client: MongoClient = MongoClient(connection_string)
        except errors.ConnectionFailure as error:
            logger.error(error)

        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

        if _motor_available:
            self.async_client = AsyncIOMotorClient(connection_string)
            self.async_db = self.async_client[database_name]
            self.async_collection = self.async_db[collection_name]
        else:
            self.async_client = None
            self.async_db = None
            self.async_collection = None

        if create_index:
            index_kwargs = index_kwargs or {}
            self.collection.create_index(self.session_id_key, **index_kwargs)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from MongoDB"""
        try:
            if self.history_size is None:
                cursor = self.collection.find({self.session_id_key: self.session_id})
            else:
                skip_count = max(
                    0,
                    self.collection.count_documents(
                        {self.session_id_key: self.session_id}
                    )
                    - self.history_size,
                )
                cursor = self.collection.find(
                    {self.session_id_key: self.session_id}, skip=skip_count
                )
        except errors.OperationFailure as error:
            logger.error(error)
            cursor = []

        if cursor:
            items = [json.loads(document[self.history_key]) for document in cursor]
        else:
            items = []

        messages = messages_from_dict(items)
        return messages

    async def aget_messages(self) -> List[BaseMessage]:
        """Async version of getting messages from MongoDB"""
        if not _motor_available:
            logger.warning(
                "Motor library is not installed. "
                "Using `run_in_executor` for aget_messages, "
                "which may be less efficient."
            )
            return await super().aget_messages()

        if self.history_size is None:
            cursor = self.async_collection.find({self.session_id_key: self.session_id})
        else:
            total_count = await self.async_collection.count_documents(
                {self.session_id_key: self.session_id}
            )
            skip_count = max(
                0,
                total_count - self.history_size,
            )
            cursor = self.async_collection.find(
                {self.session_id_key: self.session_id}, skip=skip_count
            )

        items = []
        async for document in cursor:
            items.append(json.loads(document[self.history_key]))

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

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Async add a list of messages to MongoDB"""
        if not _motor_available:
            logger.warning(
                "Motor library is not installed. "
                "Using `run_in_executor` for aadd_messages, "
                "which may be less efficient."
            )
            return await super().aadd_messages(messages)

        documents = [
            {
                self.session_id_key: self.session_id,
                self.history_key: json.dumps(message_to_dict(message)),
            }
            for message in messages
        ]
        await self.async_collection.insert_many(documents)

    def clear(self) -> None:
        """Clear session memory from MongoDB"""
        try:
            self.collection.delete_many({self.session_id_key: self.session_id})
        except errors.WriteError as err:
            logger.error(err)

    async def aclear(self) -> None:
        """Async clear session memory from MongoDB"""
        if not _motor_available:
            logger.warning(
                "Motor library is not installed. "
                "Using `run_in_executor` for aclear, which may be less efficient."
            )
            return await super().aclear()

        await self.async_collection.delete_many({self.session_id_key: self.session_id})
