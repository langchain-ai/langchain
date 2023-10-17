from typing import Any, Dict, List

from langchain.graphs.graph_document import GraphDocument
from langchain.graphs.neo4j_graph import Neo4jGraph

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


class FalkorDBGraph(Neo4jGraph):
    """FalkorDB wrapper for graph operations.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.
    """

    def __init__(
        self, database: str, host: str = "localhost", port: int = 6379
    ) -> None:
        """Create a new FalkorDB graph wrapper instance."""
        try:
            import redis
            from redis.commands.graph import Graph
        except ImportError:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        driver = redis.Redis(host=host, port=port)
        self._graph = Graph(driver, database)
        self.schema: str = ""
        self.structured_schema: Dict[str, Any] = {}

        try:
            self.refresh_schema()
        except Exception as e:
            raise ValueError(f"Could not refresh schema. Error: {e}")

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
            raise ValueError("Generated Cypher Statement is not valid\n" f"{e}")

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
