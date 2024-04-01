"""Client for persisting chat message history in a Postgres database."""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict

if TYPE_CHECKING:
    import asyncpg
    import psycopg2

logger = logging.getLogger(__name__)


def _create_table_query(table_name: str) -> str:
    """Make a SQL query to create a table."""
    if not re.match(r"^\w+$", table_name):
        raise ValueError(
            "Invalid table name. Table name must contain only alphanumeric "
            "characters and underscores."
        )
    return f"""CREATE TABLE IF NOT EXISTS {table_name} (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    message JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);"""


def _create_delete_query(table_name: str, session_id: str) -> str:
    """Make a SQL query to delete messages for a given session."""
    if not re.match(r"^\w+$", table_name):
        raise ValueError(
            "Invalid table name. Table name must contain only alphanumeric "
            "characters and underscores."
        )
    return f"DELETE FROM {table_name} WHERE session_id = %s;"


class PostgresChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        table_name: str,
        *,
        sync_connection: Optional[psycopg2.extensions.connection] = None,
        async_connection: Optional[asyncpg.connection.Connection] = None,
    ) -> None:
        """Client for persisting chat message history in a Postgres database,

        This client provides support for both sync and async via
        psycopg2 and asyncpg, respectively. It can be initialized with a DSN string,
        or with an existing connection.

        Args:
            session_id: The session ID to use for the chat message history.
            table_name: The name of the database table to use.
            sync_connection: An existing psycopg2 database client.
            async_connection: An existing asyncpg database client.

        Note:
            Must specify one of sync_connection or async_connection.
        """
        if not sync_connection and not async_connection:
            raise ValueError("Must provide sync_connectino or async_connection")
        self._connection = sync_connection
        self._aconnection = async_connection
        self._session_id = session_id

        if not re.match(r"^\w+$", table_name):
            raise ValueError(
                "Invalid table name. Table name must contain only alphanumeric "
                "characters and underscores."
            )
        self._table_name = table_name

    @classmethod
    async def acreate_schema(
        cls, connection: Optional[asyncpg.connection.Connection], table_name: str
    ) -> None:
        """Create the table schema in the database.

        Attention:
            The schema has been creates as a classmethod ratther than an instance
            method because it should be possible to create the schema without having
            to instantiate the class.

        Args:
            connection: The asyncpg connection to the database.
            table_name: The name of the table to create.
        """
        query = _create_table_query(table_name)
        await connection.execute(query)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the chat message history."""
        if self._aconnection is None:
            raise ValueError(
                "Please initialize the PostgresChatMessageHistory "
                "with an async connection or use the sync add_messages method instead."
            )

        # Bulk insert the messages into the table
        query = f"""INSERT INTO {self.table_name} (session_id, message) VALUES %s;"""
        values = [
            (self.session_id, json.dumps(message.to_dict())) for message in messages
        ]
        await self._aconnection.executemany(query, values)

    def get_messages(self) -> List[BaseMessage]:
        """Retrieve messages from the chat message history."""
        if self._connection is None:
            raise ValueError(
                "Please initialize the PostgresChatMessageHistory "
                "with a sync connection or use the async get_messages method instead."
            )

        query = (
            f"SELECT message FROM {self.table_name} WHERE session_id = %s ORDER BY id;"
        )
        self._connection.execute(query, (self.session_id,))
        items = [record["message"] for record in self._connection.fetchall()]
        messages = messages_from_dict(items)
        return messages

    def aget_messages(self) -> List[BaseMessage]:
        """Retrieve messages from the chat message history."""
        if self._aconnection is None:
            raise ValueError(
                "Please initialize the PostgresChatMessageHistory "
                "with an async connection or use the sync get_messages method instead."
            )

        query = (
            f"SELECT message FROM {self.table_name} WHERE session_id = %s ORDER BY id;"
        )
        self._aconnection.execute(query, (self.session_id,))
        items = [record["message"] for record in self._aconnection.fetchall()]
        messages = messages_from_dict(items)
        return messages

    @property
    def messages(self) -> List[BaseMessage]:
        """The abstraction required a property."""
        return self.get_messages()

    def clear(self) -> None:
        """Clear the chat message history for the given session."""
        if self._connection is None:
            raise ValueError(
                "Please initialize the PostgresChatMessageHistory "
                "with a sync connection or use the async clear method instead."
            )

        query = f"DELETE FROM {self.table_name} WHERE session_id = %s;"
        self._connection.execute(query, (self.session_id,))
        self._connection.commit()

    async def aclear(self) -> None:
        """Clear the chat message history for the given session."""
        if self._aconnection is None:
            raise ValueError(
                "Please initialize the PostgresChatMessageHistory "
                "with an async connection or use the sync clear method instead."
            )

        query = f"DELETE FROM {self.table_name} WHERE session_id = %s;"
        await self._aconnection.execute(query, (self.session_id))
        await self._aconnection.commit()
