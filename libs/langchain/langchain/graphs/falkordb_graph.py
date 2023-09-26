from typing import Any, Dict, List

from langchain.graphs.graph_document import GraphDocument
from langchain.graphs.graph_store import GraphStore

node_properties_query = """
MATCH (n)
UNWIND labels(n) as l 
UNWIND keys(n) as p WITH l, 
collect(distinct p) AS props 
RETURN {labels:l, properties: props} AS output
"""

rel_properties_query = """
MATCH ()-[r]->()
UNWIND type(r) as t
UNWIND keys(r) as p WITH t,
collect(distinct p) AS props 
RETURN {type:t, properties:props} AS output
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
        node_properties = self.query(node_properties_query)
        rel_properties = self.query(rel_properties_query)
        relationships = self.query(rel_query)

        self.structured_schema = {
            "node_props": {
                el[0]["labels"]: el[0]["properties"] for el in node_properties
            },
            "rel_props": {el[0]["type"]: el[0]["properties"] for el in rel_properties},
            "relationships": relationships,
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
                        # f"{include_docs_query if include_source else ''}"
                        f"MERGE (n:{node.type} {{id:'{node.id}'}}) "
                        "SET n += $properties "
                        # f"{'MERGE (d)-[:MENTIONS]->(n) ' if include_source else ''}"
                        "RETURN distinct 'done' AS result"
                    ),
                    {
                        "properties": node.properties,
                        "document": document.source.__dict__,
                    },
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
                    {
                        "properties": rel.properties,
                        "document": document.source.__dict__,
                    },
                )
