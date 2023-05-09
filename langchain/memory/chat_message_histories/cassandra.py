import json
import logging
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

DEFAULT_KEYSPACE_NAME = "chat_history"
DEFAULT_TABLE_NAME = "message_store"
DEFAULT_USERNAME = "cassandra"
DEFAULT_PASSWORD = "cassandra"
DEFAULT_PORT = 9042


class CassandraChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in Cassandra.
    Args:
        contact_points: list of ips to connect to Cassandra cluster
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
        port: port to connect to Cassandra cluster
        username: username to connect to Cassandra cluster
        password: password to connect to Cassandra cluster
        keyspace_name: name of the keyspace to use
        table_name: name of the table to use
    """

    def __init__(
        self,
        contact_points: List[str],
        session_id: str,
        port: int = DEFAULT_PORT,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
        keyspace_name: str = DEFAULT_KEYSPACE_NAME,
        table_name: str = DEFAULT_TABLE_NAME,
    ):
        from cassandra.cluster import Cluster, PlainTextAuthProvider

        self.contact_points = contact_points
        self.session_id = session_id
        self.port = port
        self.username = username
        self.password = password
        self.keyspace_name = keyspace_name
        self.table_name = table_name

        try:
            self.cluster: Cluster = Cluster(
                contact_points,
                port=port,
                auth_provider=PlainTextAuthProvider(
                    username=self.username, password=self.password
                ),
            )
            self.session = self.cluster.connect()
        except Exception as error:
            logger.error(error)
            self.session = None

        self._prepare_cassandra()

    def _prepare_cassandra(self) -> None:
        """Create the keyspace and table if they don't exist yet"""

        try:
            self.session.execute(
                f"""CREATE KEYSPACE IF NOT EXISTS 
                {self.keyspace_name} WITH REPLICATION = 
                {{ 'class' : 'SimpleStrategy', 'replication_factor' : 1 }};"""
            )
        except Exception as error:
            logger.error(error)

        self.session.set_keyspace(self.keyspace_name)

        try:
            self.session.execute(
                f"""CREATE TABLE IF NOT EXISTS 
                {self.table_name} (id UUID, session_id varchar, 
                history text,  PRIMARY KEY ((session_id), id) );"""
            )
        except Exception as error:
            logger.error(error)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Cassandra"""
        try:
            rows = self.session.execute(
                f"""SELECT * FROM {self.table_name}
                WHERE session_id = '{self.session_id}' ;"""
            )
        except Exception as error:
            logger.error(error)
            rows = None

        if rows:
            items = [json.loads(row.history) for row in rows]
        else:
            items = []

        messages = messages_from_dict(items)

        return messages

    def add_user_message(self, message: str) -> None:
        self.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.append(AIMessage(content=message))

    def append(self, message: BaseMessage) -> None:
        """Append the message to the record in Cassandra"""

        import uuid

        try:
            self.session.execute(
                """INSERT INTO message_store
                (id, session_id, history) VALUES (%s, %s, %s);""",
                (uuid.uuid4(), self.session_id, json.dumps(_message_to_dict(message))),
            )
        except Exception as error:
            logger.error(error)

    def clear(self) -> None:
        """Clear session memory from Cassandra"""

        try:
            self.session.execute(
                f"DELETE FROM {self.table_name} WHERE session_id = '{self.session_id}';"
            )
        except Exception as error:
            logger.error(error)

    def __del__(self) -> None:
        if self.session:
            self.session.shutdown()
        if self.cluster:
            self.cluster.shutdown()
