from typing import Any, Dict, List

from langchain.graphs.graph_document import GraphDocument

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
UNWIND other AS other_node
RETURN {start: label, type: property, end: toString(other_node)} AS output
"""


class Neo4jGraph:
    """Neo4j wrapper for graph operations.

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
        self, url: str, username: str, password: str, database: str = "neo4j"
    ) -> None:
        """Create a new Neo4j graph wrapper instance."""
        try:
            import neo4j
        except ImportError:
            raise ValueError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )

        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        self._database = database
        self.schema: str = ""
        self.structured_schema: Dict[str, Any] = {}
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
        # Set schema
        try:
            self.refresh_schema()
        except neo4j.exceptions.ClientError:
            raise ValueError(
                "Could not use APOC procedures. "
                "Please ensure the APOC plugin is installed in Neo4j and that "
                "'apoc.meta.data()' is allowed in Neo4j configuration "
            )

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Neo4j database."""
        from neo4j.exceptions import CypherSyntaxError

        with self._driver.session(database=self._database) as session:
            try:
                data = session.run(query, params)
                return [r.data() for r in data]
            except CypherSyntaxError as e:
                raise ValueError(f"Generated Cypher Statement is not valid\n{e}")

    def refresh_schema(self) -> None:
        """
        Refreshes the Neo4j graph schema information.
        """
        node_properties = [el["output"] for el in self.query(node_properties_query)]
        rel_properties = [el["output"] for el in self.query(rel_properties_query)]
        relationships = [el["output"] for el in self.query(rel_query)]

        self.structured_schema = {
            "node_props": {el["labels"]: el["properties"] for el in node_properties},
            "rel_props": {el["type"]: el["properties"] for el in rel_properties},
            "relationships": relationships,
        }
        self.schema = f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {rel_properties}
        The relationships are the following:
        {[f"(:{el['start']})-[:{el['type']}]->(:{el['end']})" for el in relationships]}
        """

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """
        Take GraphDocument as input as uses it to construct a graph.
        """
        for document in graph_documents:
            include_docs_query = (
                "CREATE (d:Document) "
                "SET d.text = $document.page_content "
                "SET d += $document.metadata "
                "WITH d "
            )
            # Import nodes
            self.query(
                (
                    f"{include_docs_query if include_source else ''}"
                    "UNWIND $data AS row "
                    "CALL apoc.merge.node([row.type], {id: row.id}, "
                    "row.properties, {}) YIELD node "
                    f"{'MERGE (d)-[:MENTIONS]->(node) ' if include_source else ''}"
                    "RETURN distinct 'done' AS result"
                ),
                {
                    "data": [el.__dict__ for el in document.nodes],
                    "document": document.source.__dict__,
                },
            )
            # Import relationships
            self.query(
                "UNWIND $data AS row "
                "CALL apoc.merge.node([row.source_label], {id: row.source},"
                "{}, {}) YIELD node as source "
                "CALL apoc.merge.node([row.target_label], {id: row.target},"
                "{}, {}) YIELD node as target "
                "CALL apoc.merge.relationship(source, row.type, "
                "{}, row.properties, target) YIELD rel "
                "RETURN distinct 'done'",
                {
                    "data": [
                        {
                            "source": el.source.id,
                            "source_label": el.source.type,
                            "target": el.target.id,
                            "target_label": el.target.type,
                            "type": el.type.replace(" ", "_").upper(),
                            "properties": el.properties,
                        }
                        for el in document.relationships
                    ]
                },
            )
