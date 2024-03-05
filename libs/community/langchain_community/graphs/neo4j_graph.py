from typing import Any, Dict, List, Optional

from langchain_core.utils import get_from_env

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore

BASE_ENTITY_LABEL = "__Entity__"

node_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "node" 
  AND NOT label IN [$BASE_ENTITY_LABEL]
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
WITH * WHERE NOT label IN [$BASE_ENTITY_LABEL] 
    AND NOT other_node IN [$BASE_ENTITY_LABEL]
RETURN {start: label, type: property, end: toString(other_node)} AS output
"""

include_docs_query = (
    "CREATE (d:Document) "
    "SET d.text = $document.page_content "
    "SET d += $document.metadata "
    "WITH d "
)


def value_sanitize(d: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize the input dictionary.

    Sanitizes the input dictionary by removing embedding-like values,
    lists with more than 128 elements, that are mostly irrelevant for
    generating answers in a LLM context. These properties, if left in
    results, can occupy significant context space and detract from
    the LLM's performance by introducing unnecessary noise and cost.
    """
    LIST_LIMIT = 128
    # Create a new dictionary to avoid changing size during iteration
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # Recurse to handle nested dictionaries
            new_dict[key] = value_sanitize(value)
        elif isinstance(value, list):
            # check if it has less than LIST_LIMIT values
            if len(value) < LIST_LIMIT:
                # if value is a list, check if it contains dictionaries to clean
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_list.append(value_sanitize(item))
                    else:
                        cleaned_list.append(item)
                new_dict[key] = cleaned_list  # type: ignore[assignment]
        else:
            new_dict[key] = value
    return new_dict


def _get_node_import_query(baseEntityLabel: bool, include_source: bool) -> str:
    if baseEntityLabel:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.id}}) "
            "SET source += row.properties "
            f"{'MERGE (d)-[:MENTIONS]->(source) ' if include_source else ''}"
            "WITH source, row "
            "CALL apoc.create.addLabels( source, [row.type] ) YIELD node "
            "RETURN distinct 'done' AS result"
        )
    else:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.type], {id: row.id}, "
            "row.properties, {}) YIELD node "
            f"{'MERGE (d)-[:MENTIONS]->(node) ' if include_source else ''}"
            "RETURN distinct 'done' AS result"
        )


def _get_rel_import_query(baseEntityLabel: bool) -> str:
    if baseEntityLabel:
        return (
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.source}}) "
            f"MERGE (target:`{BASE_ENTITY_LABEL}` {{id: row.target}}) "
            "WITH source, target, row "
            "CALL apoc.merge.relationship(source, row.type, "
            "{}, row.properties, target) YIELD rel "
            "RETURN distinct 'done'"
        )
    else:
        return (
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.source_label], {id: row.source},"
            "{}, {}) YIELD node as source "
            "CALL apoc.merge.node([row.target_label], {id: row.target},"
            "{}, {}) YIELD node as target "
            "CALL apoc.merge.relationship(source, row.type, "
            "{}, row.properties, target) YIELD rel "
            "RETURN distinct 'done'"
        )


class Neo4jGraph(GraphStore):
    """Neo4j database wrapper for various graph operations.

    Parameters:
    url (Optional[str]): The URL of the Neo4j database server.
    username (Optional[str]): The username for database authentication.
    password (Optional[str]): The password for database authentication.
    database (str): The name of the database to connect to. Default is 'neo4j'.
    timeout (Optional[float]): The timeout for transactions in seconds.
            Useful for terminating long-running queries.
            By default, there is no timeout set.
    sanitize (bool): A flag to indicate whether to remove lists with
            more than 128 elements from results. Useful for removing
            embedding-like properties from database responses. Default is False.

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
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        timeout: Optional[float] = None,
        sanitize: bool = False,
    ) -> None:
        """Create a new Neo4j graph wrapper instance."""
        try:
            import neo4j
        except ImportError:
            raise ValueError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )

        url = get_from_env("url", "NEO4J_URI", url)
        username = get_from_env("username", "NEO4J_USERNAME", username)
        password = get_from_env("password", "NEO4J_PASSWORD", password)
        database = get_from_env("database", "NEO4J_DATABASE", database)

        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        self._database = database
        self.timeout = timeout
        self.sanitize = sanitize
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
        except neo4j.exceptions.ClientError as e:
            if e.code == "Neo.ClientError.Procedure.ProcedureNotFound":
                raise ValueError(
                    "Could not use APOC procedures. "
                    "Please ensure the APOC plugin is installed in Neo4j and that "
                    "'apoc.meta.data()' is allowed in Neo4j configuration "
                )
            raise e

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph"""
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the structured schema of the Graph"""
        return self.structured_schema

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Neo4j database."""
        from neo4j import Query
        from neo4j.exceptions import CypherSyntaxError

        with self._driver.session(database=self._database) as session:
            try:
                data = session.run(Query(text=query, timeout=self.timeout), params)
                json_data = [r.data() for r in data]
                if self.sanitize:
                    json_data = [value_sanitize(el) for el in json_data]
                return json_data
            except CypherSyntaxError as e:
                raise ValueError(f"Generated Cypher Statement is not valid\n{e}")

    def refresh_schema(self) -> None:
        """
        Refreshes the Neo4j graph schema information.
        """
        from neo4j.exceptions import ClientError

        node_properties = [
            el["output"]
            for el in self.query(
                node_properties_query, params={"BASE_ENTITY_LABEL": BASE_ENTITY_LABEL}
            )
        ]
        rel_properties = [
            el["output"]
            for el in self.query(
                rel_properties_query, params={"BASE_ENTITY_LABEL": BASE_ENTITY_LABEL}
            )
        ]
        relationships = [
            el["output"]
            for el in self.query(
                rel_query, params={"BASE_ENTITY_LABEL": BASE_ENTITY_LABEL}
            )
        ]

        # Get constraints & indexes
        try:
            constraint = self.query("SHOW CONSTRAINTS")
            index = self.query("SHOW INDEXES YIELD *")
        except (
            ClientError
        ):  # Read-only user might not have access to schema information
            constraint = []
            index = []

        self.structured_schema = {
            "node_props": {el["labels"]: el["properties"] for el in node_properties},
            "rel_props": {el["type"]: el["properties"] for el in rel_properties},
            "relationships": relationships,
            "metadata": {"constraint": constraint, "index": index},
        }

        # Format node properties
        formatted_node_props = []
        for el in node_properties:
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in el["properties"]]
            )
            formatted_node_props.append(f"{el['labels']} {{{props_str}}}")

        # Format relationship properties
        formatted_rel_props = []
        for el in rel_properties:
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in el["properties"]]
            )
            formatted_rel_props.append(f"{el['type']} {{{props_str}}}")

        # Format relationships
        formatted_rels = [
            f"(:{el['start']})-[:{el['type']}]->(:{el['end']})" for el in relationships
        ]

        self.schema = "\n".join(
            [
                "Node properties are the following:",
                ",".join(formatted_node_props),
                "Relationship properties are the following:",
                ",".join(formatted_rel_props),
                "The relationships are the following:",
                ",".join(formatted_rels),
            ]
        )

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        baseEntityLabel: bool = False,
    ) -> None:
        """
        This method constructs nodes and relationships in the graph based on the
        provided GraphDocument objects.

        Parameters:
        - graph_documents (List[GraphDocument]): A list of GraphDocument objects
        that contain the nodes and relationships to be added to the graph. Each
        GraphDocument should encapsulate the structure of part of the graph,
        including nodes, relationships, and the source document information.
        - include_source (bool, optional): If True, stores the source document
        and links it to nodes in the graph using the MENTIONS relationship.
        This is useful for tracing back the origin of data. Defaults to False.
        - baseEntityLabel (bool, optional): If True, each newly created node
        gets a secondary __Entity__ label, which is indexed and improves import
        speed and performance. Defaults to False.
        """
        if baseEntityLabel:  # Check if constraint already exists
            constraint_exists = any(
                [
                    el["labelsOrTypes"] == [BASE_ENTITY_LABEL]
                    and el["properties"] == ["id"]
                    for el in self.structured_schema.get("metadata", {}).get(
                        "constraint"
                    )
                ]
            )
            if not constraint_exists:
                # Create constraint
                self.query(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (b:{BASE_ENTITY_LABEL}) "
                    "REQUIRE b.id IS UNIQUE;"
                )
                self.refresh_schema()  # Refresh constraint information

        node_import_query = _get_node_import_query(baseEntityLabel, include_source)
        rel_import_query = _get_rel_import_query(baseEntityLabel)
        for document in graph_documents:
            # Import nodes
            self.query(
                node_import_query,
                {
                    "data": [el.__dict__ for el in document.nodes],
                    "document": document.source.__dict__,
                },
            )
            # Import relationships
            self.query(
                rel_import_query,
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
