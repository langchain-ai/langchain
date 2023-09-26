import json
import logging
from datetime import datetime
from typing import List

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict

logger = logging.getLogger(__name__)

DEFAULT_CONNECTION_STRING = "postgresql://postgres:mypassword@localhost/chat_history"


class PostgresChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Postgres database."""

    def __init__(
        self,
        session_id: str,
        connection_string: str = DEFAULT_CONNECTION_STRING,
        table_name: str = "message_store",
        descending_time: bool = True,
        limit: int = -1,
    ):
        import psycopg
        from psycopg.rows import dict_row

        try:
            self.connection = psycopg.connect(connection_string)
            self.cursor = self.connection.cursor(row_factory=dict_row)
            self.descending_time = descending_time
            self.limit = limit
        except psycopg.OperationalError as error:
            logger.error(error)

        self.session_id = session_id
        self.table_name = table_name

        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        create_table_query = f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            message JSONB NOT NULL,
            created_at TIMESTAMPTZ
        );"""
        self.cursor.execute(create_table_query)
        self.connection.commit()

    @property
    def messages_order_by_time(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from PostgreSQL order by time"""
        order = "desc" if self.descending_time else "asc"
        params: List = [self.session_id]
        if self.limit > 0:
            limit = " limit %s"
            params.append(self.limit)
        else:
            limit = ""
        query = f"SELECT message FROM {self.table_name} WHERE session_id = %s ORDER BY created_at {order}{limit};"  # noqa: E501
        self.cursor.execute(query, tuple(params))
        items = [record["message"] for record in self.cursor.fetchall()]
        messages = messages_from_dict(items)
        return messages

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

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = sql.SQL(
            "INSERT INTO {} (session_id, message, created_at) VALUES (%s, %s, %s);"
        ).format(sql.Identifier(self.table_name))
        self.cursor.execute(
            query, (self.session_id, json.dumps(_message_to_dict(message)), created_at)
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