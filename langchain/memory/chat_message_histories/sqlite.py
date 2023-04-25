import json
import logging
import sqlite3
from typing import List

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./history.sqlite3"


def dict_row(cursor: sqlite3.Cursor, row: tuple) -> dict:
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        db_path: str = DEFAULT_DB_PATH,
        table_name: str = "message_store",
    ):
        self.db_path = db_path
        try:
            self.connection = sqlite3.connect(db_path)
            self.connection.row_factory = dict_row
            self.cursor = self.connection.cursor()
        except sqlite3.OperationalError as error:
            logger.error(error)

        self.session_id = session_id
        self.table_name = table_name

        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        create_table_query = f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            message TEXT NOT NULL
        );"""
        self.cursor.execute(create_table_query)
        self.connection.commit()

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from SQLite"""
        query = f"SELECT message FROM {self.table_name} WHERE session_id = ?;"
        self.cursor.execute(query, (self.session_id,))
        items = [json.loads(record["message"]) for record in self.cursor.fetchall()]
        messages = messages_from_dict(items)
        return messages

    def add_user_message(self, message: str) -> None:
        self.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.append(AIMessage(content=message))

    def append(self, message: BaseMessage) -> None:
        """Append the message to the record in SQLite"""

        query = f"INSERT INTO {self.table_name} (session_id, message) VALUES (?, ?);"
        self.cursor.execute(
            query, (self.session_id, json.dumps(_message_to_dict(message)))
        )
        self.connection.commit()

    def clear(self) -> None:
        """Clear session memory from SQLite"""
        query = f"DELETE FROM {self.table_name} WHERE session_id = ?;"
        self.cursor.execute(query, (self.session_id,))
        self.connection.commit()

    def __del__(self) -> None:
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
