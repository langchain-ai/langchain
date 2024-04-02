"""Client for persisting chat message history in a Postgres database.

This client provides support for both sync and async via psycopg 3.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import List, Optional, Sequence

import psycopg
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from psycopg import sql

logger = logging.getLogger(__name__)


def _create_table_and_index(table_name: str) -> List[sql.Composed]:
    """Make a SQL query to create a table."""
    index_name = f"idx_{table_name}_session_id"
    statements = [
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                session_id UUID NOT NULL,
                message JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        ).format(table_name=sql.Identifier(table_name)),
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} (session_id);
            """
        ).format(
            table_name=sql.Identifier(table_name), index_name=sql.Identifier(index_name)
        ),
    ]
    return statements


def _get_messages_query(table_name: str) -> sql.Composed:
    """Make a SQL query to get messages for a given session."""
    return sql.SQL(
        "SELECT message "
        "FROM {table_name} "
        "WHERE session_id = %(session_id)s "
        "ORDER BY id;"
    ).format(table_name=sql.Identifier(table_name))


def _delete_by_session_id_query(table_name: str) -> sql.Composed:
    """Make a SQL query to delete messages for a given session."""
    return sql.SQL(
        "DELETE FROM {table_name} WHERE session_id = %(session_id)s;"
    ).format(table_name=sql.Identifier(table_name))


def _delete_table_query(table_name: str) -> sql.Composed:
    """Make a SQL query to delete a table."""
    return sql.SQL("DROP TABLE IF EXISTS {table_name};").format(
        table_name=sql.Identifier(table_name)
    )


def _insert_message_query(table_name: str) -> sql.Composed:
    """Make a SQL query to insert a message."""
    return sql.SQL(
        "INSERT INTO {table_name} (session_id, message) VALUES (%s, %s)"
    ).format(table_name=sql.Identifier(table_name))


class PostgresChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        table_name: str,
        session_id: str,
        /,
        *,
        sync_connection: Optional[psycopg.Connection] = None,
        async_connection: Optional[psycopg.AsyncConnection] = None,
    ) -> None:
        """Client for persisting chat message history in a Postgres database,

        This client provides support for both sync and async via psycopg >=3.

        The client can create schema in the database and provides methods to
        add messages, get messages, and clear the chat message history.

        The schema has the following columns:

        - id: A serial primary key.
        - session_id: The session ID for the chat message history.
        - message: The JSONB message content.
        - created_at: The timestamp of when the message was created.

        Messages are retrieved for a given session_id and are sorted by
        the id (which should be increasing monotonically), and correspond
        to the order in which the messages were added to the history.

        The "created_at" column is not returned by the interface, but
        has been added for the schema so the information is available in the database.

        A session_id can be used to separate different chat histories in the same table,
        the session_id should be provided when initializing the client.

        This chat history client takes in a psycopg connection object (either
        Connection or AsyncConnection) and uses it to interact with the database.

        This design allows to reuse the underlying connection object across
        multiple instantiations of this class, making instantiation fast.

        This chat history client is designed for prototyping applications that
        involve chat and are based on Postgres.

        As your application grows, you will likely need to extend the schema to
        handle more complex queries. For example, a chat application
        may involve multiple tables like a user table, a table for storing
        chat sessions / conversations, and this table for storing chat messages
        for a given session. The application will require access to additional
        endpoints like deleting messages by user id, listing conversations by
        user id or ordering them based on last message time, etc.

        Feel free to adapt this implementation to suit your application's needs.

        Args:
            session_id: The session ID to use for the chat message history
            table_name: The name of the database table to use
            sync_connection: An existing psycopg connection instance
            async_connection: An existing psycopg async connection instance

        Usage:
            - Use the create_schema or acreate_schema method to set up the table
              schema in the database.
            - Initialize the class with the appropriate session ID, table name,
              and database connection.
            - Add messages to the database using add_messages or aadd_messages.
            - Retrieve messages with get_messages or aget_messages.
            - Clear the session history with clear or aclear when needed.

        Note:
            - At least one of sync_connection or async_connection must be provided.

        Examples:

        .. code-block:: python

            import uuid

            from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
            from langchain_postgres import PostgresChatMessageHistory
            import psycopg

            # Establish a synchronous connection to the database
            # (or use psycopg.AsyncConnection for async)
            sync_connection = psycopg2.connect(conn_info)

            # Create the table schema (only needs to be done once)
            table_name = "chat_history"
            PostgresChatMessageHistory.create_schema(sync_connection, table_name)

            session_id = str(uuid.uuid4())

            # Initialize the chat history manager
            chat_history = PostgresChatMessageHistory(
                table_name,
                session_id,
                sync_connection=sync_connection
            )

            # Add messages to the chat history
            chat_history.add_messages([
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ])

            print(chat_history.messages)
        """
        if not sync_connection and not async_connection:
            raise ValueError("Must provide sync_connection or async_connection")

        self._connection = sync_connection
        self._aconnection = async_connection

        # Validate that session id is a UUID
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise ValueError(
                f"Invalid session id. Session id must be a valid UUID. Got {session_id}"
            )

        self._session_id = session_id

        if not re.match(r"^\w+$", table_name):
            raise ValueError(
                "Invalid table name. Table name must contain only alphanumeric "
                "characters and underscores."
            )
        self._table_name = table_name

    @staticmethod
    def create_schema(
        connection: psycopg.Connection,
        table_name: str,
        /,
    ) -> None:
        """Create the table schema in the database and create relevant indexes."""
        queries = _create_table_and_index(table_name)
        logger.info("Creating schema for table %s", table_name)
        with connection.cursor() as cursor:
            for query in queries:
                cursor.execute(query)
        connection.commit()

    @staticmethod
    async def acreate_schema(
        connection: psycopg.AsyncConnection, table_name: str, /
    ) -> None:
        """Create the table schema in the database and create relevant indexes."""
        queries = _create_table_and_index(table_name)
        logger.info("Creating schema for table %s", table_name)
        async with connection.cursor() as cur:
            for query in queries:
                await cur.execute(query)
        await connection.commit()

    @staticmethod
    def drop_table(connection: psycopg.Connection, table_name: str, /) -> None:
        """Delete the table schema in the database.

        WARNING:
            This will delete the given table from the database including
            all the database in the table and the schema of the table.

        Args:
            connection: The database connection.
            table_name: The name of the table to create.
        """

        query = _delete_table_query(table_name)
        logger.info("Dropping table %s", table_name)
        with connection.cursor() as cursor:
            cursor.execute(query)
        connection.commit()

    @staticmethod
    async def adrop_table(
        connection: psycopg.AsyncConnection, table_name: str, /
    ) -> None:
        """Delete the table schema in the database.

        WARNING:
            This will delete the given table from the database including
            all the database in the table and the schema of the table.

        Args:
            connection: Async database connection.
            table_name: The name of the table to create.
        """
        query = _delete_table_query(table_name)
        logger.info("Dropping table %s", table_name)

        async with connection.cursor() as acur:
            await acur.execute(query)
        await connection.commit()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the chat message history."""
        if self._connection is None:
            raise ValueError(
                "Please initialize the PostgresChatMessageHistory "
                "with a sync connection or use the aadd_messages method instead."
            )

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
            raise ValueError(
                "Please initialize the PostgresChatMessageHistory "
                "with an async connection or use the sync add_messages method instead."
            )

        values = [
            (self._session_id, json.dumps(message_to_dict(message)))
            for message in messages
        ]

        query = _insert_message_query(self._table_name)
        async with self._aconnection.cursor() as cursor:
            await cursor.executemany(query, values)
        await self._aconnection.commit()

    def get_messages(self) -> List[BaseMessage]:
        """Retrieve messages from the chat message history."""
        if self._connection is None:
            raise ValueError(
                "Please initialize the PostgresChatMessageHistory "
                "with a sync connection or use the async aget_messages method instead."
            )

        query = _get_messages_query(self._table_name)

        with self._connection.cursor() as cursor:
            cursor.execute(query, {"session_id": self._session_id})
            items = [record[0] for record in cursor.fetchall()]

        messages = messages_from_dict(items)
        return messages

    async def aget_messages(self) -> List[BaseMessage]:
        """Retrieve messages from the chat message history."""
        if self._aconnection is None:
            raise ValueError(
                "Please initialize the PostgresChatMessageHistory "
                "with an async connection or use the sync get_messages method instead."
            )

        query = _get_messages_query(self._table_name)
        async with self._aconnection.cursor() as cursor:
            await cursor.execute(query, {"session_id": self._session_id})
            items = [record[0] for record in await cursor.fetchall()]

        messages = messages_from_dict(items)
        return messages

    @property  # type: ignore[override]
    def messages(self) -> List[BaseMessage]:
        """The abstraction required a property."""
        return self.get_messages()

    def clear(self) -> None:
        """Clear the chat message history for the GIVEN session."""
        if self._connection is None:
            raise ValueError(
                "Please initialize the PostgresChatMessageHistory "
                "with a sync connection or use the async clear method instead."
            )

        query = _delete_by_session_id_query(self._table_name)
        with self._connection.cursor() as cursor:
            cursor.execute(query, {"session_id": self._session_id})
        self._connection.commit()

    async def aclear(self) -> None:
        """Clear the chat message history for the GIVEN session."""
        if self._aconnection is None:
            raise ValueError(
                "Please initialize the PostgresChatMessageHistory "
                "with an async connection or use the sync clear method instead."
            )

        query = _delete_by_session_id_query(self._table_name)
        async with self._aconnection.cursor() as cursor:
            await cursor.execute(query, {"session_id": self._session_id})
        await self._aconnection.commit()
