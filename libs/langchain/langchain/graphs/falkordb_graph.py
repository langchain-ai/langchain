from typing import Any, Dict, List

from langchain.graphs.graph_document import GraphDocument
from langchain.graphs.graph_store import GraphStore

node_properties_query = """
MATCH (n)
UNWIND labels(n) as l
UNWIND keys(n) as p
RETURN {label:l, properties: collect(distinct p)} AS output
"""

rel_properties_query = """
MATCH ()-[r]->()
UNWIND keys(r) as p
RETURN {type:type(r), properties: collect(distinct p)} AS output
"""

rel_query = """
MATCH (n)-[r]->(m)
WITH labels(n)[0] AS src, labels(m)[0] AS dst, type(r) AS type
RETURN DISTINCT "(:" + src + ")-[:" + type + "]->(:" + dst + ")" AS output
"""


class FalkorDBGraph(GraphStore):
    """FalkorDB wrapper for graph operations."""

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

        self._driver = redis.Redis(host=host, port=port)
        self._graph = Graph(self._driver, database)

        try:
            self.refresh_schema()
        except Exception as e:
            raise ValueError(f"Could not refresh schema. Error: {e}")

    @property
    def get_schema(self) -> str:
        """Returns the schema of the FalkorDB database"""
        return self.schema

    def refresh_schema(self) -> None:
        """Refreshes the schema of the FalkorDB database"""
        self.schema = (
            f"Node properties: {self.query(node_properties_query)}\n"
            f"Relationships properties: {self.query(rel_properties_query)}\n"
            f"Relationships: {self.query(rel_query)}\n"
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
                props = " ".join(
                    "SET n.{0}='{1}'".format(k, v.replace("'", "\\'"))
                    if isinstance(v, str)
                    else "SET n.{0}={1}".format(k, v)
                    for k, v in node.properties.items()
                )

                self.query(
                    (
                        # f"{include_docs_query if include_source else ''}"
                        f"MERGE (n:{node.type} {{id:'{node.id}'}}) "
                        f"{props} "
                        # f"{'MERGE (d)-[:MENTIONS]->(n) ' if include_source else ''}"
                        "RETURN distinct 'done' AS result"
                    )
                )

            # Import relationships
            for rel in document.relationships:
                props = " ".join(
                    "SET r.{0}='{1}'".format(k, v.replace("'", "\\'"))
                    if isinstance(v, str)
                    else "SET r.{0}={1}".format(k, v)
                    for k, v in rel.properties.items()
                )

                self.query(
                    (
                        f"MATCH (a:{rel.source.type} {{id:'{rel.source.id}'}}), "
                        f"(b:{rel.target.type} {{id:'{rel.target.id}'}}) "
                        f"MERGE (a)-[r:{(rel.type.replace(' ', '_').upper())}]->(b) "
                        f"{props} "
                        "RETURN distinct 'done' AS result"
                    )
                )
