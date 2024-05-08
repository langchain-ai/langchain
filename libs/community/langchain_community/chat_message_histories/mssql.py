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


def _delete_by_session_id_query(table_name: str) -> str:
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
        """Client for persisting chat message history in a MSSQL database,

        This client provides support for both sync and async via pyodbc and aioodbc.

        In order to use, you must have an ODBC driver manager installed.
        On windows, this is usually built in. On mac, install with brew
        install unixodbc. On linux / unix, refer to distro guides.

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

        This chat history client takes in a pyodbc or aioodbc connection object (either
        Connection or AsyncConnection) and uses it to interact with the database.

        This design allows to reuse the underlying connection object across
        multiple instantiations of this class, making instantiation fast.

        This chat history client is designed for prototyping applications that
        involve chat and are based on MSSQL.

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
            sync_connection: An existing pyodbc connection instance
            async_connection: An existing aioodbc async connection instance

        Usage:
            - Use the create_tables or acreate_tables method to set up the table
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
            from langchain_community.chat_message_historys.mssql
            import MssqlChatMessageHistory
            import pyodbc
            import aioodbc

            # Establish a synchronous connection to the database
            # (or use aioodbc.Connection for async)
            conn_info = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"
                f"SERVER={MSSQL_SERVER};"
                f"DATABASE={MSSQL_DATABASE};"
                f"UID={MSSQL_USERNAME};"
                f"PWD={MSSQL_PASSWORD};"
                f"Encrypt=no;"
                f"TrustServerCertificate=yes;"
            )


            sync_connection = pyodbc.connect(conn_info)
            # async_connection = await aioodbc.connect(dsn=conn_info)

            # Create the table schema (only needs to be done once)
            table_name = "chat_history"
            MssqlChatMessageHistory.create_tables(sync_connection, table_name)

            session_id = str(uuid.uuid4())

            # Initialize the chat history manager
            chat_history = MssqlChatMessageHistory(
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
          This will delete the given table from the database including all
          the databases in the table and the schema of the table.

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
          This will delete the given table from the database including all
          the databases in the table and the schema of the table.

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
        async with self._aconnection.cursor() as cursor:
            await cursor.execute(query, self._session_id)
            await cursor.commit()
