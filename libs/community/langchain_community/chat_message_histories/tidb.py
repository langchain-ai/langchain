import json
import logging
from datetime import datetime
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class TiDBChatMessageHistory(BaseChatMessageHistory):
    """
    Represents a chat message history stored in a TiDB database.
    """

    def __init__(
        self,
        session_id: str,
        connection_string: str,
        table_name: str = "langchain_message_store",
        earliest_time: Optional[datetime] = None,
    ):
        """
        Initializes a new instance of the TiDBChatMessageHistory class.

        Args:
            session_id (str): The ID of the chat session.
            connection_string (str): The connection string for the TiDB database.
                format: mysql+pymysql://<host>:<PASSWORD>@<host>:4000/<db>?ssl_ca=/etc/ssl/cert.pem&ssl_verify_cert=true&ssl_verify_identity=true
            table_name (str, optional): the table name to store the chat messages.
                Defaults to "langchain_message_store".
            earliest_time (Optional[datetime], optional): The earliest time to retrieve messages from.
                Defaults to None.
        """  # noqa

        self.session_id = session_id
        self.table_name = table_name
        self.earliest_time = earliest_time
        self.cache = []

        # Set up SQLAlchemy engine and session
        self.engine = create_engine(connection_string)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        self._create_table_if_not_exists()
        self._load_messages_to_cache()

    def _create_table_if_not_exists(self) -> None:
        """
        Creates a table if it does not already exist in the database.
        """

        create_table_query = text(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                message JSON NOT NULL,
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX session_idx (session_id)
            );"""
        )
        try:
            self.session.execute(create_table_query)
            self.session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Error creating table: {e}")
            self.session.rollback()

    def _load_messages_to_cache(self) -> None:
        """
        Loads messages from the database into the cache.

        This method retrieves messages from the database table. The retrieved messages
        are then stored in the cache for faster access.

        Raises:
            SQLAlchemyError: If there is an error executing the database query.

        """
        time_condition = (
            f"AND create_time >= '{self.earliest_time}'" if self.earliest_time else ""
        )
        query = text(
            f"""
            SELECT message FROM {self.table_name} 
            WHERE session_id = :session_id {time_condition} 
            ORDER BY id;
        """
        )
        try:
            result = self.session.execute(query, {"session_id": self.session_id})
            for record in result.fetchall():
                message_dict = json.loads(record[0])
                self.cache.append(messages_from_dict([message_dict])[0])
        except SQLAlchemyError as e:
            logger.error(f"Error loading messages to cache: {e}")

    @property
    def messages(self) -> List[BaseMessage]:
        """returns all messages"""
        if len(self.cache) == 0:
            self.reload_cache()
        return self.cache

    def add_message(self, message: BaseMessage) -> None:
        """adds a message to the database and cache"""
        query = text(
            f"INSERT INTO {self.table_name} (session_id, message) VALUES (:session_id, :message);"  # noqa
        )
        try:
            self.session.execute(
                query,
                {
                    "session_id": self.session_id,
                    "message": json.dumps(message_to_dict(message)),
                },
            )
            self.session.commit()
            self.cache.append(message)
        except SQLAlchemyError as e:
            logger.error(f"Error adding message: {e}")
            self.session.rollback()

    def clear(self) -> None:
        """clears all messages"""
        query = text(f"DELETE FROM {self.table_name} WHERE session_id = :session_id;")
        try:
            self.session.execute(query, {"session_id": self.session_id})
            self.session.commit()
            self.cache.clear()
        except SQLAlchemyError as e:
            logger.error(f"Error clearing messages: {e}")
            self.session.rollback()

    def reload_cache(self) -> None:
        """reloads messages from database to cache"""
        self.cache.clear()
        self._load_messages_to_cache()

    def __del__(self) -> None:
        """closes the session"""
        self.session.close()
