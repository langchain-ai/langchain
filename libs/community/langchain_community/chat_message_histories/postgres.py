import json
import logging
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

logger = logging.getLogger(__name__)


class PostgresChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Postgres database."""

    def __init__(self, session_id: str, database_config: dict, table_name: str = "message_store"):
        """
        Initializes a new instance of the chat message history with specified session ID, 
        database configuration, and table name. Attempts to establish a database connection 
        using the provided configuration settings.
        
        Args:
            session_id (str): A unique identifier for the chat session.
            database_config (dict): Configuration settings for the database connection. 
                Expected to contain the following keys: DB_HOST, DB_NAME, DB_USER, 
                DB_PASSWORD, and DB_PORT.
            table_name (str): The name of the database table to store chat messages. 
                Defaults to 'message_store'.
                
        Raises:
            psycopg2.OperationalError: If the connection to the database fails.
        """
        import psycopg2

        try:
            connection_pool = psycopg2.pool.ThreadedConnectionPool(
                host=database_config["DB_HOST"],
                dbname=database_config["DB_NAME"],
                user=database_config["DB_USER"],
                password=database_config["DB_PASSWORD"],
                port=database_config["DB_PORT"],
            )

            self.connection = connection_pool.getconn()
            self.cursor = self.connection.cursor()
        except psycopg2.OperationalError as error:
            logger.error(f"Failed to get connection from pool: {error}")

        self.session_id = session_id
        self.table_name = table_name

        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        create_table_query = f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            message JSONB NOT NULL
        );"""
        self.cursor.execute(create_table_query)
        self.connection.commit()

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from PostgreSQL"""
        query = (
            f"SELECT message FROM {self.table_name} WHERE session_id = %s ORDER BY id;"
        )
        self.cursor.execute(query, (self.session_id,))
        items = [record["message"] for record in self.cursor.fetchall()]
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in PostgreSQL"""
        from psycopg import sql

        query = sql.SQL("INSERT INTO {} (session_id, message) VALUES (%s, %s);").format(
            sql.Identifier(self.table_name)
        )
        self.cursor.execute(
            query, (self.session_id, json.dumps(message_to_dict(message)))
        )
        self.connection.commit()

    def clear(self) -> None:
        """Clear session memory from PostgreSQL"""
        query = f"DELETE FROM {self.table_name} WHERE session_id = %s;"
        self.cursor.execute(query, (self.session_id,))
        self.connection.commit()

    def __del__(self) -> None:
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
