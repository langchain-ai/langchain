import warnings
from typing import Any, Dict, List, Optional

from langchain_core._api import deprecated

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore

node_properties_query = """
MATCH (n)
WITH keys(n) as keys, labels(n) AS labels
WITH CASE WHEN keys = [] THEN [NULL] ELSE keys END AS keys, labels
UNWIND labels AS label
UNWIND keys AS key
WITH label, collect(DISTINCT key) AS keys
RETURN {label:label, keys:keys} AS output
"""

rel_properties_query = """
MATCH ()-[r]->()
WITH keys(r) as keys, type(r) AS types
WITH CASE WHEN keys = [] THEN [NULL] ELSE keys END AS keys, types 
UNWIND types AS type
UNWIND keys AS key WITH type,
collect(DISTINCT key) AS keys 
RETURN {types:type, keys:keys} AS output
"""

rel_query = """
MATCH (n)-[r]->(m)
UNWIND labels(n) as src_label
UNWIND labels(m) as dst_label
UNWIND type(r) as rel_type
RETURN DISTINCT {start: src_label, type: rel_type, end: dst_label} AS output
"""


class FalkorDBGraph(GraphStore):
    """FalkorDB wrapper for graph operations.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(
        self,
        database: str,
        host: str = "localhost",
        port: int = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssl: bool = False,
    ) -> None:
        """Create a new FalkorDB graph wrapper instance."""
        try:
            self.__init_falkordb_connection(
                database, host, port, username, password, ssl
            )

        except ImportError:
            try:
                # Falls back to using the redis package just for backwards compatibility
                self.__init_redis_connection(
                    database, host, port, username, password, ssl
                )
            except ImportError:
                raise ImportError(
                    "Could not import falkordb python package. "
                    "Please install it with `pip install falkordb`."
                )

        self.schema: str = ""
        self.structured_schema: Dict[str, Any] = {}

        try:
            self.refresh_schema()
        except Exception as e:
            raise ValueError(f"Could not refresh schema. Error: {e}")

    def __init_falkordb_connection(
        self,
        database: str,
        host: str = "localhost",
        port: int = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssl: bool = False,
    ) -> None:
        from falkordb import FalkorDB

        try:
            self._driver = FalkorDB(
                host=host, port=port, username=username, password=password, ssl=ssl
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to FalkorDB: {e}")

        self._graph = self._driver.select_graph(database)

    @deprecated("0.0.31", alternative="__init_falkordb_connection")
    def __init_redis_connection(
        self,
        database: str,
        host: str = "localhost",
        port: int = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssl: bool = False,
    ) -> None:
        import redis
        from redis.commands.graph import Graph

        # show deprecation warning
        warnings.warn(
            "Using the redis package is deprecated. "
            "Please use the falkordb package instead, "
            "install it with `pip install falkordb`.",
            DeprecationWarning,
        )

        self._driver = redis.Redis(
            host=host, port=port, username=username, password=password, ssl=ssl
        )

        self._graph = Graph(self._driver, database)

    @property
    def get_schema(self) -> str:
        """Returns the schema of the FalkorDB database"""
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the structured schema of the Graph"""
        return self.structured_schema

    def refresh_schema(self) -> None:
        """Refreshes the schema of the FalkorDB database"""
        node_properties: List[Any] = self.query(node_properties_query)
        rel_properties: List[Any] = self.query(rel_properties_query)
        relationships: List[Any] = self.query(rel_query)

        self.structured_schema = {
            "node_props": {el[0]["label"]: el[0]["keys"] for el in node_properties},
            "rel_props": {el[0]["types"]: el[0]["keys"] for el in rel_properties},
            "relationships": [el[0] for el in relationships],
        }

        self.schema = (
            f"Node properties: {node_properties}\n"
            f"Relationships properties: {rel_properties}\n"
            f"Relationships: {relationships}\n"
        )

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query FalkorDB database."""

        try:
            data = self._graph.query(query, params)
            return data.result_set
        except Exception as e:
            raise ValueError(f"Generated Cypher Statement is not valid\n{e}")

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """
        Take GraphDocument as input as uses it to construct a graph.
        """
        for document in graph_documents:
            # Import nodes
            for node in document.nodes:
                self.query(
                    (
                        f"MERGE (n:{node.type} {{id:'{node.id}'}}) "
                        "SET n += $properties "
                        "RETURN distinct 'done' AS result"
                    ),
                    {"properties": node.properties},
                )

            # Import relationships
            for rel in document.relationships:
                self.query(
                    (
                        f"MATCH (a:{rel.source.type} {{id:'{rel.source.id}'}}), "
                        f"(b:{rel.target.type} {{id:'{rel.target.id}'}}) "
                        f"MERGE (a)-[r:{(rel.type.replace(' ', '_').upper())}]->(b) "
                        "SET r += $properties "
                        "RETURN distinct 'done' AS result"
                    ),
                    {"properties": rel.properties},
                )
