"""Cassandra-based chat message history, based on cassIO."""
from __future__ import annotations

import json
import typing
from typing import List

if typing.TYPE_CHECKING:
    from cassandra.cluster import Session

from langchain.schema import (
    BaseChatMessageHistory,
    BaseMessage,
    _message_to_dict,
    messages_from_dict,
)

DEFAULT_TABLE_NAME = "message_store"
DEFAULT_TTL_SECONDS = None


class CassandraChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in Cassandra.

    Args:
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
        session: a Cassandra `Session` object (an open DB connection)
        keyspace: name of the keyspace to use.
        table_name: name of the table to use.
        ttl_seconds: time-to-live (seconds) for automatic expiration
            of stored entries. None (default) for no expiration.
    """

    def __init__(
        self,
        session_id: str,
        session: Session,
        keyspace: str,
        table_name: str = DEFAULT_TABLE_NAME,
        ttl_seconds: int | None = DEFAULT_TTL_SECONDS,
    ) -> None:
        try:
            from cassio.history import StoredBlobHistory
        except (ImportError, ModuleNotFoundError):
            raise ValueError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )
        self.session_id = session_id
        self.ttl_seconds = ttl_seconds
        self.blob_history = StoredBlobHistory(session, keyspace, table_name)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all session messages from DB"""
        message_blobs = self.blob_history.retrieve(
            self.session_id,
        )
        items = [json.loads(message_blob) for message_blob in message_blobs]
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Write a message to the table"""
        self.blob_history.store(
            self.session_id, json.dumps(_message_to_dict(message)), self.ttl_seconds
        )

    def clear(self) -> None:
        """Clear session memory from DB"""
        self.blob_history.clear_session_id(self.session_id)
