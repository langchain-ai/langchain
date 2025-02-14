from typing import List, Optional, Union

from langchain_core._api.deprecation import deprecated
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.utils import get_from_dict_or_env

from langchain_community.graphs import Neo4jGraph


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.Neo4jChatMessageHistory",
)
class Neo4jChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Neo4j database."""

    def __init__(
        self,
        session_id: Union[str, int],
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        node_label: str = "Session",
        window: int = 3,
        *,
        graph: Optional[Neo4jGraph] = None,
    ):
        try:
            import neo4j
        except ImportError:
            raise ImportError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )

        # Make sure session id is not null
        if not session_id:
            raise ValueError("Please ensure that the session_id parameter is provided")

        # Graph object takes precedent over env or input params
        if graph:
            self._driver = graph._driver
            self._database = graph._database
        else:
            # Handle if the credentials are environment variables
            url = get_from_dict_or_env({"url": url}, "url", "NEO4J_URI")
            username = get_from_dict_or_env(
                {"username": username}, "username", "NEO4J_USERNAME"
            )
            password = get_from_dict_or_env(
                {"password": password}, "password", "NEO4J_PASSWORD"
            )
            database = get_from_dict_or_env(
                {"database": database}, "database", "NEO4J_DATABASE", "neo4j"
            )

            self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
            self._database = database
            # Verify connection
            try:
                self._driver.verify_connectivity()
            except neo4j.exceptions.ServiceUnavailable:
                raise ValueError(
                    "Could not connect to Neo4j database. "
                    "Please ensure that the url is correct"
                )
            except neo4j.exceptions.AuthError:
                raise ValueError(
                    "Could not connect to Neo4j database. "
                    "Please ensure that the username and password are correct"
                )
        self._session_id = session_id
        self._node_label = node_label
        self._window = window
        # Create session node
        self._driver.execute_query(
            f"MERGE (s:`{self._node_label}` {{id:$session_id}})",
            {"session_id": self._session_id},
        ).summary

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from Neo4j"""
        query = (
            f"MATCH (s:`{self._node_label}`)-[:LAST_MESSAGE]->(last_message) "
            "WHERE s.id = $session_id MATCH p=(last_message)<-[:NEXT*0.."
            f"{self._window * 2}]-() WITH p, length(p) AS length "
            "ORDER BY length DESC LIMIT 1 UNWIND reverse(nodes(p)) AS node "
            "RETURN {data:{content: node.content}, type:node.type} AS result"
        )
        records, _, _ = self._driver.execute_query(
            query, {"session_id": self._session_id}
        )

        messages = messages_from_dict([el["result"] for el in records])
        return messages

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError(
            "Direct assignment to 'messages' is not allowed."
            " Use the 'add_messages' instead."
        )

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Neo4j"""
        query = (
            f"MATCH (s:`{self._node_label}`) WHERE s.id = $session_id "
            "OPTIONAL MATCH (s)-[lm:LAST_MESSAGE]->(last_message) "
            "CREATE (s)-[:LAST_MESSAGE]->(new:Message) "
            "SET new += {type:$type, content:$content} "
            "WITH new, lm, last_message WHERE last_message IS NOT NULL "
            "CREATE (last_message)-[:NEXT]->(new) "
            "DELETE lm"
        )
        self._driver.execute_query(
            query,
            {
                "type": message.type,
                "content": message.content,
                "session_id": self._session_id,
            },
        ).summary

    def clear(self) -> None:
        """Clear session memory from Neo4j"""
        query = (
            f"MATCH (s:`{self._node_label}`)-[:LAST_MESSAGE]->(last_message) "
            "WHERE s.id = $session_id MATCH p=(last_message)<-[:NEXT]-() "
            "WITH p, length(p) AS length ORDER BY length DESC LIMIT 1 "
            "UNWIND nodes(p) as node DETACH DELETE node;"
        )
        self._driver.execute_query(query, {"session_id": self._session_id}).summary

    def __del__(self) -> None:
        if self._driver:
            self._driver.close()
