from typing import List, Dict, Any

node_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {labels: nodeLabels, properties: properties} AS output

"""

rel_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {type: nodeLabels, properties: properties} AS output
"""

rel_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE type = "RELATIONSHIP" AND elementType = "node"
RETURN "(:" + label + ")-[:" + property + "]->(:" + toString(other[0]) + ")" AS output
"""


class Neo4jGraph:
    """Neo4j wrapper for entity graph operations."""

    def __init__(self, url: str, username: str, password: str) -> None:
        """Create a new graph."""
        try:
            import neo4j
        except ImportError:
            raise ValueError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )

        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        # Set schema
        try:
            self.refresh_schema()
        except Exception as e:
            raise ValueError(
                e
            )

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Neo4j database
        """
        return self.schema


    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Neo4j database."""
        with self._driver.session() as session:
            data = session.run(query, params)
            # Hard limit of 200 results
            return [r.data() for r in data][:200]
    
    def refresh_schema(self):
        """
        Refreshes the Neo4j graph schema information.
        """
        node_properties = self.query(node_properties_query)
        relationships_properties = self.query(rel_properties_query)
        relationships = self.query(rel_query)

        self.schema = f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {relationships_properties}
        The relationships are the following:
        {relationships}
        """

