import logging
from hashlib import md5
from typing import Any, Dict, List, Optional

from langchain_core.utils import get_from_dict_or_env

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_community.graphs.graph_store import GraphStore

logger = logging.getLogger(__name__)


BASE_ENTITY_LABEL = "__Entity__"

SCHEMA_QUERY = """
SHOW SCHEMA INFO
"""

NODE_PROPERTIES_QUERY = """
CALL schema.node_type_properties()
YIELD nodeType AS label, propertyName AS property, propertyTypes AS type
WITH label AS nodeLabels, collect({key: property, types: type}) AS properties
RETURN {labels: nodeLabels, properties: properties} AS output
"""

REL_QUERY = """
MATCH (n)-[e]->(m)
WITH DISTINCT
    labels(n) AS start_node_labels,
    type(e) AS rel_type,
    labels(m) AS end_node_labels,
    e,
    keys(e) AS properties
UNWIND CASE WHEN size(properties) > 0 THEN properties ELSE [null] END AS prop
WITH
    start_node_labels,
    rel_type,
    end_node_labels,
    CASE WHEN prop IS NULL THEN [] ELSE [prop, valueType(e[prop])] END AS property_info
RETURN
    start_node_labels,
    rel_type,
    end_node_labels,
    COLLECT(DISTINCT CASE 
    WHEN property_info <> [] 
    THEN property_info 
    ELSE null END) AS properties_info
"""

NODE_IMPORT_QUERY = """
UNWIND $data AS row
CALL merge.node(row.label, row.properties, {}, {}) 
YIELD node 
RETURN distinct 'done' AS result
"""

REL_NODES_IMPORT_QUERY = """
UNWIND $data AS row
MERGE (source {id: row.source_id})
MERGE (target {id: row.target_id})
RETURN distinct 'done' AS result
"""

REL_IMPORT_QUERY = """
UNWIND $data AS row
MATCH (source {id: row.source_id})
MATCH (target {id: row.target_id})
WITH source, target, row
CALL merge.relationship(source, row.type, {}, {}, target, {})
YIELD rel
RETURN distinct 'done' AS result
"""

INCLUDE_DOCS_QUERY = """
MERGE (d:Document {id:$document.metadata.id})
SET d.content = $document.page_content
SET d += $document.metadata
RETURN distinct 'done' AS result
"""

INCLUDE_DOCS_SOURCE_QUERY = """
UNWIND $data AS row
MATCH (source {id: row.source_id}), (d:Document {id: $document.metadata.id})
MERGE (d)-[:MENTIONS]->(source)
RETURN distinct 'done' AS result
"""

NODE_PROPS_TEXT = """
Node labels and properties (name and type) are:
"""

REL_PROPS_TEXT = """
Relationship labels and properties are:
"""

REL_TEXT = """
Nodes are connected with the following relationships:
"""


def get_schema_subset(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "edges": [
            {
                "end_node_labels": edge["end_node_labels"],
                "properties": [
                    {
                        "key": prop["key"],
                        "types": [
                            {"type": type_item["type"].lower()}
                            for type_item in prop["types"]
                        ],
                    }
                    for prop in edge["properties"]
                ],
                "start_node_labels": edge["start_node_labels"],
                "type": edge["type"],
            }
            for edge in data["edges"]
        ],
        "nodes": [
            {
                "labels": node["labels"],
                "properties": [
                    {
                        "key": prop["key"],
                        "types": [
                            {"type": type_item["type"].lower()}
                            for type_item in prop["types"]
                        ],
                    }
                    for prop in node["properties"]
                ],
            }
            for node in data["nodes"]
        ],
    }


def get_reformated_schema(
    nodes: List[Dict[str, Any]], rels: List[Dict[str, Any]]
) -> Dict[str, Any]:
    return {
        "edges": [
            {
                "end_node_labels": rel["end_node_labels"],
                "properties": [
                    {"key": prop[0], "types": [{"type": prop[1].lower()}]}
                    for prop in rel["properties_info"]
                ],
                "start_node_labels": rel["start_node_labels"],
                "type": rel["rel_type"],
            }
            for rel in rels
        ],
        "nodes": [
            {
                "labels": [_remove_backticks(node["labels"])[1:]],
                "properties": [
                    {
                        "key": prop["key"],
                        "types": [
                            {"type": type_item.lower()} for type_item in prop["types"]
                        ],
                    }
                    for prop in node["properties"]
                    if node["properties"][0]["key"] != ""
                ],
            }
            for node in nodes
        ],
    }


def transform_schema_to_text(schema: Dict[str, Any]) -> str:
    node_props_data = ""
    rel_props_data = ""
    rel_data = ""

    for node in schema["nodes"]:
        node_props_data += f"- labels: (:{':'.join(node['labels'])})\n"
        if node["properties"] == []:
            continue
        node_props_data += "  properties:\n"
        for prop in node["properties"]:
            prop_types_str = " or ".join(
                {prop_types["type"] for prop_types in prop["types"]}
            )
            node_props_data += f"    - {prop['key']}: {prop_types_str}\n"

    for rel in schema["edges"]:
        rel_type = rel["type"]
        start_labels = ":".join(rel["start_node_labels"])
        end_labels = ":".join(rel["end_node_labels"])
        rel_data += f"(:{start_labels})-[:{rel_type}]->(:{end_labels})\n"

        if rel["properties"] == []:
            continue

        rel_props_data += f"- labels: {rel_type}\n  properties:\n"
        for prop in rel["properties"]:
            prop_types_str = " or ".join(
                {prop_types["type"].lower() for prop_types in prop["types"]}
            )
            rel_props_data += f"    - {prop['key']}: {prop_types_str}\n"

    return "".join(
        [
            NODE_PROPS_TEXT + node_props_data if node_props_data else "",
            REL_PROPS_TEXT + rel_props_data if rel_props_data else "",
            REL_TEXT + rel_data if rel_data else "",
        ]
    )


def _remove_backticks(text: str) -> str:
    return text.replace("`", "")


def _transform_nodes(nodes: list[Node], baseEntityLabel: bool) -> List[dict]:
    transformed_nodes = []
    for node in nodes:
        properties_dict = node.properties | {"id": node.id}
        label = (
            [_remove_backticks(node.type), BASE_ENTITY_LABEL]
            if baseEntityLabel
            else [_remove_backticks(node.type)]
        )
        node_dict = {"label": label, "properties": properties_dict}
        transformed_nodes.append(node_dict)
    return transformed_nodes


def _transform_relationships(
    relationships: list[Relationship], baseEntityLabel: bool
) -> List[dict]:
    transformed_relationships = []
    for rel in relationships:
        rel_dict = {
            "type": _remove_backticks(rel.type),
            "source_label": (
                [BASE_ENTITY_LABEL]
                if baseEntityLabel
                else [_remove_backticks(rel.source.type)]
            ),
            "source_id": rel.source.id,
            "target_label": (
                [BASE_ENTITY_LABEL]
                if baseEntityLabel
                else [_remove_backticks(rel.target.type)]
            ),
            "target_id": rel.target.id,
        }
        transformed_relationships.append(rel_dict)
    return transformed_relationships


class MemgraphGraph(GraphStore):
    """Memgraph wrapper for graph operations.

    Parameters:
    url (Optional[str]): The URL of the Memgraph database server.
    username (Optional[str]): The username for database authentication.
    password (Optional[str]): The password for database authentication.
    database (str): The name of the database to connect to. Default is 'memgraph'.
    refresh_schema (bool): A flag whether to refresh schema information
    at initialization. Default is True.
    driver_config (Dict): Configuration passed to Neo4j Driver.

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
        database: Optional[str] = None,
        refresh_schema: bool = True,
        *,
        driver_config: Optional[Dict] = None,
    ) -> None:
        """Create a new Memgraph graph wrapper instance."""
        try:
            import neo4j
        except ImportError:
            raise ImportError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )

        url = get_from_dict_or_env({"url": url}, "url", "MEMGRAPH_URI")

        # if username and password are "", assume auth is disabled
        if username == "" and password == "":
            auth = None
        else:
            username = get_from_dict_or_env(
                {"username": username},
                "username",
                "MEMGRAPH_USERNAME",
            )
            password = get_from_dict_or_env(
                {"password": password},
                "password",
                "MEMGRAPH_PASSWORD",
            )
            auth = (username, password)
        database = get_from_dict_or_env(
            {"database": database}, "database", "MEMGRAPH_DATABASE", "memgraph"
        )

        self._driver = neo4j.GraphDatabase.driver(
            url, auth=auth, **(driver_config or {})
        )

        self._database = database
        self.schema: str = ""
        self.structured_schema: Dict[str, Any] = {}

        # Verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Memgraph database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Memgraph database. "
                "Please ensure that the username and password are correct"
            )

        # Set schema
        if refresh_schema:
            try:
                self.refresh_schema()
            except neo4j.exceptions.ClientError as e:
                raise e

    def close(self) -> None:
        if self._driver:
            logger.info("Closing the driver connection.")
            self._driver.close()
            self._driver = None

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph database"""
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the structured schema of the Graph database"""
        return self.structured_schema

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query the graph.

        Args:
            query (str): The Cypher query to execute.
            params (dict): The parameters to pass to the query.

        Returns:
            List[Dict[str, Any]]: The list of dictionaries containing the query results.
        """
        from neo4j.exceptions import Neo4jError

        try:
            data, _, _ = self._driver.execute_query(
                query,
                database_=self._database,
                parameters_=params,
            )
            json_data = [r.data() for r in data]
            return json_data
        except Neo4jError as e:
            if not (
                (
                    (  # isCallInTransactionError
                        e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                        or e.code
                        == "Neo.DatabaseError.Transaction.TransactionStartFailed"
                    )
                    and "in an implicit transaction" in e.message
                )
                or (  # isPeriodicCommitError
                    e.code == "Neo.ClientError.Statement.SemanticError"
                    and (
                        "in an open transaction is not possible" in e.message
                        or "tried to execute in an explicit transaction" in e.message
                    )
                )
                or (
                    e.code == "Memgraph.ClientError.MemgraphError.MemgraphError"
                    and ("in multicommand transactions" in e.message)
                )
                or (
                    e.code == "Memgraph.ClientError.MemgraphError.MemgraphError"
                    and "SchemaInfo disabled" in e.message
                )
            ):
                raise

        # fallback to allow implicit transactions
        with self._driver.session(database=self._database) as session:
            data = session.run(query, params)
            json_data = [r.data() for r in data]
            return json_data

    def refresh_schema(self) -> None:
        """
        Refreshes the Memgraph graph schema information.
        """
        import ast

        from neo4j.exceptions import Neo4jError

        # leave schema empty if db is empty
        if self.query("MATCH (n) RETURN n LIMIT 1") == []:
            return

        # first try with SHOW SCHEMA INFO
        try:
            result = self.query(SCHEMA_QUERY)[0].get("schema")
            if result is not None and isinstance(result, (str, ast.AST)):
                schema_result = ast.literal_eval(result)
            else:
                schema_result = result
            assert schema_result is not None
            structured_schema = get_schema_subset(schema_result)
            self.structured_schema = structured_schema
            self.schema = transform_schema_to_text(structured_schema)
            return
        except Neo4jError as e:
            if (
                e.code == "Memgraph.ClientError.MemgraphError.MemgraphError"
                and "SchemaInfo disabled" in e.message
            ):
                logger.info(
                    "Schema generation with SHOW SCHEMA INFO query failed. "
                    "Set --schema-info-enabled=true to use SHOW SCHEMA INFO query. "
                    "Falling back to alternative queries."
                )

        # fallback on Cypher without SHOW SCHEMA INFO
        nodes = [query["output"] for query in self.query(NODE_PROPERTIES_QUERY)]
        rels = self.query(REL_QUERY)

        structured_schema = get_reformated_schema(nodes, rels)
        self.structured_schema = structured_schema
        self.schema = transform_schema_to_text(structured_schema)

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        baseEntityLabel: bool = False,
    ) -> None:
        """
        Take GraphDocument as input as uses it to construct a graph in Memgraph.

        Parameters:
        - graph_documents (List[GraphDocument]): A list of GraphDocument objects
        that contain the nodes and relationships to be added to the graph. Each
        GraphDocument should encapsulate the structure of part of the graph,
        including nodes, relationships, and the source document information.
        - include_source (bool, optional): If True, stores the source document
        and links it to nodes in the graph using the MENTIONS relationship.
        This is useful for tracing back the origin of data. Merges source
        documents based on the `id` property from the source document metadata
        if available; otherwise it calculates the MD5 hash of `page_content`
        for merging process. Defaults to False.
        - baseEntityLabel (bool, optional): If True, each newly created node
        gets a secondary __Entity__ label, which is indexed and improves import
        speed and performance. Defaults to False.
        """

        if baseEntityLabel:
            self.query(
                f"CREATE CONSTRAINT ON (b:{BASE_ENTITY_LABEL}) "
                "ASSERT b.id IS UNIQUE;"
            )
            self.query(f"CREATE INDEX ON :{BASE_ENTITY_LABEL}(id);")
            self.query(f"CREATE INDEX ON :{BASE_ENTITY_LABEL};")

        for document in graph_documents:
            if include_source:
                if not document.source.metadata.get("id"):
                    document.source.metadata["id"] = md5(
                        document.source.page_content.encode("utf-8")
                    ).hexdigest()

                self.query(INCLUDE_DOCS_QUERY, {"document": document.source.__dict__})

            self.query(
                NODE_IMPORT_QUERY,
                {"data": _transform_nodes(document.nodes, baseEntityLabel)},
            )

            rel_data = _transform_relationships(document.relationships, baseEntityLabel)
            self.query(
                REL_NODES_IMPORT_QUERY,
                {"data": rel_data},
            )
            self.query(
                REL_IMPORT_QUERY,
                {"data": rel_data},
            )

            if include_source:
                self.query(
                    INCLUDE_DOCS_SOURCE_QUERY,
                    {"data": rel_data, "document": document.source.__dict__},
                )
        self.refresh_schema()
