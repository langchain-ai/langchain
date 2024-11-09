import os
from typing import List, Optional, Union

import redis.exceptions
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from langchain_community.graphs import FalkorDBGraph


class FalkorDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Falkor database.

    This class handles storing and retrieving chat messages in a FalkorDB database.
    It creates a session and stores messages in a message chain, maintaining a link
    between subsequent messages.

    Args:
        session_id (Union[str, int]): The session ID for storing and retrieving messages
                                also the name of the database.
        username (Optional[str]): Username for authenticating with FalkorDB.
        password (Optional[str]): Password for authenticating with FalkorDB.
        host (str): Host where the FalkorDB is running. Defaults to 'localhost'.
        port (int): Port number where the FalkorDB is running. Defaults to 6379.
        node_label (str): Label for the session node
                        in the graph. Defaults to "Session".
        window (int): The number of messages to retrieve when querying
                        the history. Defaults to 3.
        ssl (bool): Whether to use SSL for connecting
                    to the database. Defaults to False.
        graph (Optional[FalkorDBGraph]): Optionally provide an existing
                    FalkorDBGraph object for connecting.

    Example:
        .. code-block:: python
            from langchain_community.chat_message_histories import (
            FalkorDBChatMessageHistory
            )

            history = FalkorDBChatMessageHistory(
                session_id="1234",
                host="localhost",
                port=6379,
            )
            history.add_message(HumanMessage(content="Hello!"))
    """

    def __init__(
        self,
        session_id: Union[str, int],
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        node_label: str = "Session",
        window: int = 3,
        ssl: bool = False,
        *,
        graph: Optional[FalkorDBGraph] = None,
    ) -> None:
        """
        Initialize the FalkorDBChatMessageHistory
        class with the session and connection details.
        """
        try:
            import falkordb
        except ImportError:
            raise ImportError(
                "Could not import falkordb python package."
                "Please install it with `pip install falkordb`."
            )

        if not session_id:
            raise ValueError("Please ensure that the session_id parameter is provided.")

        if graph:
            self._database = graph._graph
            self._driver = graph._driver
        else:
            self._host = host
            self._port = port
            self._username = username or os.environ.get("FALKORDB_USERNAME")
            self._password = password or os.environ.get("FALKORDB_PASSWORD")
            self._ssl = ssl

            try:
                self._driver = falkordb.FalkorDB(
                    host=self._host,
                    port=self._port,
                    username=self._username,
                    password=self._password,
                    ssl=self._ssl,
                )
            except redis.exceptions.ConnectionError:
                raise ValueError(
                    "Could not connect to FalkorDB database. "
                    "Please ensure that the host and port are correct."
                )
            except redis.exceptions.AuthenticationError:
                raise ValueError(
                    "Could not connect to FalkorDB database. "
                    "Please ensure that the username and password are correct."
                )

        self._database = self._driver.select_graph(session_id)
        self._session_id = session_id
        self._node_label = node_label
        self._window = window

        self._database.query(
            f"MERGE (s:{self._node_label} {{id:$session_id}})",
            {"session_id": self._session_id},
        )

        try:
            self._database.create_node_vector_index(
                f"{self._node_label}", "id", dim=5, similarity_function="cosine"
            )
        except Exception as e:
            if "already indexed" in str(e):
                raise ValueError(f"{self._node_label} has already been indexed")

    def _process_records(self, records: list) -> List[BaseMessage]:
        """Process the records from FalkorDB and convert them into BaseMessage objects.

        Args:
            records (list): The raw records fetched from the FalkorDB query.

        Returns:
            List[BaseMessage]: A list of `BaseMessage` objects.
        """
        # Explicitly set messages as a list of BaseMessage
        messages: List[BaseMessage] = []

        for record in records:
            content = record[0].get("data", {}).get("content", "")
            message_type = record[0].get("type", "").lower()

            # Append the correct message type to the list
            if message_type == "human":
                messages.append(
                    HumanMessage(
                        content=content, additional_kwargs={}, response_metadata={}
                    )
                )
            elif message_type == "ai":
                messages.append(
                    AIMessage(
                        content=content, additional_kwargs={}, response_metadata={}
                    )
                )
            else:
                raise ValueError(f"Unknown message type: {message_type}")

        return messages

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from FalkorDB for the session.

        Returns:
            List[BaseMessage]: A list of messages in the current session.
        """
        query = (
            f"MATCH (s:{self._node_label})-[:LAST_MESSAGE]->(last_message) "
            "MATCH p=(last_message)<-[:NEXT*0.."
            f"{self._window*2}]-() WITH p, length(p) AS length "
            "ORDER BY length DESC LIMIT 1 UNWIND reverse(nodes(p)) AS node "
            "RETURN {data:{content: node.content}, type:node.type} AS result"
        )

        records = self._database.query(query).result_set

        messages = self._process_records(records)
        return messages

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        """Block direct assignment to 'messages' to prevent misuse."""
        raise NotImplementedError(
            "Direct assignment to 'messages' is not allowed."
            " Use the 'add_message' method instead."
        )

    def add_message(self, message: BaseMessage) -> None:
        """Append a message to the session in FalkorDB.

        Args:
            message (BaseMessage): The message object to add to the session.
        """
        create_query = (
            f"MATCH (s:{self._node_label}) "
            "CREATE (new:Message {type: $type, content: $content}) "
            "WITH s, new "
            "OPTIONAL MATCH (s)-[lm:LAST_MESSAGE]->(last_message:Message) "
            "FOREACH (_ IN CASE WHEN last_message IS NULL THEN [] ELSE [1] END | "
            "  MERGE (last_message)-[:NEXT]->(new)) "
            "MERGE (s)-[:LAST_MESSAGE]->(new) "
        )

        self._database.query(
            create_query,
            {
                "type": message.type,
                "content": message.content,
            },
        )

    def clear(self) -> None:
        """Clear all messages from the session in FalkorDB.

        Deletes all messages linked to the session and resets the message history.

        Raises:
            ValueError: If there is an issue with the query or the session.
        """
        query = (
            f"MATCH (s:{self._node_label})-[:LAST_MESSAGE|NEXT*0..]->(m:Message) "
            "WITH m DELETE m"
        )
        self._database.query(query)
