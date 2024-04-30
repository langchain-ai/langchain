from __future__ import annotations

import json
import re
import uuid
from typing import List, Optional, Sequence

import aioodbc
import pyodbc
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)


def _create_table_and_index(table_name: str) -> List[str]:
    """Make a SQL query to create a table and index."""
    index_name_session_id = f"idx_{table_name}_session_id"
    statements = [
        """
    CREATE TABLE {table_name} (
      id INT IDENTITY(1,1) PRIMARY KEY,
      session_id UNIQUEIDENTIFIER NOT NULL,
      message NVARCHAR(MAX) NOT NULL,
      create_at DATETIMEOFFSET NOT NULL DEFAULT GETDATE()
    )
    """.format(table_name=table_name),
        """
    CREATE INDEX {index_name_session_id} ON {table_name} (session_id)
    """.format(index_name_session_id=index_name_session_id, table_name=table_name),
    ]
    return statements


def _get_messages_query(table_name: str) -> str:
    """Make an MSSQL query to get messages for a given session."""
    return """
    SELECT message
    FROM {table_name}
    WHERE session_id = ?
    ORDER BY create_at
    """.format(table_name=table_name)


def _delete_by_session_id_query(table_name: str):
    """Make a MSSQL query to delete messages for a given session"""
    return """
    DELETE FROM {table_name} WHERE session_id = ?
  """.format(table_name=table_name)


def _delete_table_query(table_name: str) -> str:
    """Make an MSSQL query to delete a table"""
    return """
      DROP TABLE IF EXISTS {table_name}
    """.format(table_name=table_name)


def _insert_message_query(table_name: str) -> str:
    """Make an MSSQL query to insert a message into the table."""
    return """
    INSERT INTO {table_name} (session_id, message)
    VALUES (?, ?, ?)
    """.format(table_name=table_name)


class MssqlChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        table_name: str,
        session_id: str,
        /,
        *,
        sync_connection: Optional[pyodbc.Connection] = None,
        async_connection: Optional[aioodbc.Connection] = None,
    ) -> None:
        """Client for persisting chat message history in a MSSQL database."""
        if not sync_connection and not async_connection:
            raise ValueError("Must provide either a sync or async connection.")

        self._connection = sync_connection
        self._aconnection = async_connection

        # Validate that session id is an UUID
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise ValueError("Session id must be a valid UUID.")

        self._session_id = session_id

        if not re.match(r"^\w+$", table_name):
            raise ValueError("Table name must be alphanumeric.")

        self._table_name = table_name

    @staticmethod
    def create_tables(connection: pyodbc.Connection, table_name: str, /) -> None:
        """Create the table schema in the database and create relevant indexes."""
        queries = _create_table_and_index(table_name)
        with connection.cursor() as cursor:
            for query in queries:
                cursor.execute(query)
                cursor.commit()

    @staticmethod
    async def acreate_tables(
        connection: aioodbc.Connection, table_name: str, /
    ) -> None:
        """Create the table schema in the database and create relevant indexes."""
        queries = _create_table_and_index(table_name)
        async with connection.cursor() as cursor:
            for query in queries:
                await cursor.execute(query)
                await cursor.commit()

    @staticmethod
    def drop_table(connection: pyodbc.Connection, table_name: str, /) -> None:
        """Delete the table schema in the database.

        WARNING:
          This will delete the given table from the database including all the databases in the table and the schema of the table.

        Args:
          connection: The database engine.
          table_name: the name of the table to drop
        """

        query = _delete_table_query(table_name)
        with connection.cursor() as cursor:
            cursor.execute(query)
            cursor.commit()

    @staticmethod
    async def adrop_table(connection: aioodbc.Connection, table_name: str, /) -> None:
        """Delete the table schema in the database.

        WARNING:
          This will delete the given table from the database including all the databases in the table and the schema of the table.

        Args:
          connection: The database engine.
          table_name: the name of the table to drop
        """
        query = _delete_table_query(table_name)
        async with connection.cursor() as cursor:
            await cursor.execute(query)
            await cursor.commit()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the chat message history."""
        if self._connection is None:
            raise ValueError("No connection to the database.")
        values = [
            (self._session_id, json.dumps(message_to_dict(message)))
            for message in messages
        ]
        query = _insert_message_query(self._table_name)
        with self._connection.cursor() as cursor:
            cursor.executemany(query, values)
            self._connection.commit()

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the chat message history."""
        if self._aconnection is None:
            raise ValueError("No connection to the database.")

        values = [
            (self._session_id, json.dumps(message_to_dict(message)))
            for message in messages
        ]
        query = _insert_message_query(self._table_name)
        async with self._aconnection.cursor() as cursor:
            await cursor.executemany(query, values)
            await cursor.commit()

    def get_messages(self) -> List[BaseMessage]:
        """Retrieve messages from the chat message history."""
        if self._connection is None:
            raise ValueError("No connection to the database.")

        query = _get_messages_query(self._table_name)

        with self._connection.cursor() as cursor:
            cursor.execute(query, self._session_id)
            result = cursor.fetchall()
            items = [json.loads(record[0]) for record in result]
        messages = messages_from_dict(items)
        return messages

    async def aget_messages(self) -> List[BaseMessage]:
        """Retrieve messages from the chat message history."""
        if self._aconnection is None:
            raise ValueError("No connection to the database.")

        query = _get_messages_query(self._table_name)

        async with self._aconnection.cursor() as cursor:
            await cursor.execute(query, self._session_id)
            result = await cursor.fetchall()
            items = [json.loads(record[0]) for record in result]
            messages = messages_from_dict(items)
        return messages

    @property
    def messages(self) -> List[BaseMessage]:
        """The messages in the chat message history."""
        return self.get_messages()

    def clear(self) -> None:
        """Clear the chat message history for the given session."""
        if self._connection is None:
            raise ValueError("No connection to the database.")

        query = _delete_by_session_id_query(self._table_name)
        with self._connection.cursor() as cursor:
            cursor.execute(query, self._session_id)
            cursor.commit()

    async def aclear(self) -> None:
        """Clear the chat message history for the given session."""
        if self._aconnection is None:
            raise ValueError("No connection to the database.")

        query = _delete_by_session_id_query(self._table_name)
        cursor = await self._aconnection.cursor()
        await cursor.execute(query, self._session_id)
        await cursor.commit()
        await cursor.close()
