from hashlib import md5
from typing import Any, Dict, List, Optional

from langchain_core.utils import get_from_dict_or_env

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore

BASE_ENTITY_LABEL = "__Entity__"
EXCLUDED_LABELS = ["_Bloom_Perspective_", "_Bloom_Scene_"]
EXCLUDED_RELS = ["_Bloom_HAS_SCENE_"]
EXHAUSTIVE_SEARCH_LIMIT = 10000
LIST_LIMIT = 128
# Threshold for returning all available prop values in graph schema
DISTINCT_VALUE_LIMIT = 10

node_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "node" 
  AND NOT label IN $EXCLUDED_LABELS
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {labels: nodeLabels, properties: properties} AS output

"""

rel_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
      AND NOT label in $EXCLUDED_LABELS
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {type: nodeLabels, properties: properties} AS output
"""

rel_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE type = "RELATIONSHIP" AND elementType = "node"
UNWIND other AS other_node
WITH * WHERE NOT label IN $EXCLUDED_LABELS
    AND NOT other_node IN $EXCLUDED_LABELS
RETURN {start: label, type: property, end: toString(other_node)} AS output
"""

include_docs_query = (
    "MERGE (d:Document {id:$document.metadata.id}) "
    "SET d.text = $document.page_content "
    "SET d += $document.metadata "
    "WITH d "
)


def clean_string_values(text: str) -> str:
    """Clean string values for schema.

    Cleans the input text by replacing newline and carriage return characters.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    return text.replace("\n", " ").replace("\r", " ")


def value_sanitize(d: Any) -> Any:
    """Sanitize the input dictionary or list.

    Sanitizes the input by removing embedding-like values,
    lists with more than 128 elements, that are mostly irrelevant for
    generating answers in a LLM context. These properties, if left in
    results, can occupy significant context space and detract from
    the LLM's performance by introducing unnecessary noise and cost.

    Args:
        d (Any): The input dictionary or list to sanitize.

    Returns:
        Any: The sanitized dictionary or list.
    """
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                sanitized_value = value_sanitize(value)
                if (
                    sanitized_value is not None
                ):  # Check if the sanitized value is not None
                    new_dict[key] = sanitized_value
            elif isinstance(value, list):
                if len(value) < LIST_LIMIT:
                    sanitized_value = value_sanitize(value)
                    if (
                        sanitized_value is not None
                    ):  # Check if the sanitized value is not None
                        new_dict[key] = sanitized_value
                # Do not include the key if the list is oversized
            else:
                new_dict[key] = value
        return new_dict
    elif isinstance(d, list):
        if len(d) < LIST_LIMIT:
            return [
                value_sanitize(item) for item in d if value_sanitize(item) is not None
            ]
        else:
            return None
    else:
        return d


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


def _format_schema(schema: Dict, is_enhanced: bool) -> str:
    formatted_node_props = []
    formatted_rel_props = []
    if is_enhanced:
        # Enhanced formatting for nodes
        for node_type, properties in schema["node_props"].items():
            formatted_node_props.append(f"- **{node_type}**")
            for prop in properties:
                example = ""
                if prop["type"] == "STRING" and prop.get("values"):
                    if prop.get("distinct_count", 11) > DISTINCT_VALUE_LIMIT:
                        example = (
                            f'Example: "{clean_string_values(prop["values"][0])}"'
                            if prop["values"]
                            else ""
                        )
                    else:  # If less than 10 possible values return all
                        example = (
                            (
                                "Available options: "
                                f'{[clean_string_values(el) for el in prop["values"]]}'
                            )
                            if prop["values"]
                            else ""
                        )

                elif prop["type"] in [
                    "INTEGER",
                    "FLOAT",
                    "DATE",
                    "DATE_TIME",
                    "LOCAL_DATE_TIME",
                ]:
                    if prop.get("min") is not None:
                        example = f'Min: {prop["min"]}, Max: {prop["max"]}'
                    else:
                        example = (
                            f'Example: "{prop["values"][0]}"'
                            if prop.get("values")
                            else ""
                        )
                elif prop["type"] == "LIST":
                    # Skip embeddings
                    if not prop.get("min_size") or prop["min_size"] > LIST_LIMIT:
                        continue
                    example = (
                        f'Min Size: {prop["min_size"]}, Max Size: {prop["max_size"]}'
                    )
                formatted_node_props.append(
                    f"  - `{prop['property']}`: {prop['type']} {example}"
                )

        # Enhanced formatting for relationships
        for rel_type, properties in schema["rel_props"].items():
            formatted_rel_props.append(f"- **{rel_type}**")
            for prop in properties:
                example = ""
                if prop["type"] == "STRING":
                    if prop.get("distinct_count", 11) > DISTINCT_VALUE_LIMIT:
                        example = (
                            f'Example: "{clean_string_values(prop["values"][0])}"'
                            if prop["values"]
                            else ""
                        )
                    else:  # If less than 10 possible values return all
                        example = (
                            (
                                "Available options: "
                                f'{[clean_string_values(el) for el in prop["values"]]}'
                            )
                            if prop["values"]
                            else ""
                        )
                elif prop["type"] in [
                    "INTEGER",
                    "FLOAT",
                    "DATE",
                    "DATE_TIME",
                    "LOCAL_DATE_TIME",
                ]:
                    if prop.get("min"):  # If we have min/max
                        example = f'Min: {prop["min"]}, Max:  {prop["max"]}'
                    else:  # return a single value
                        example = (
                            f'Example: "{prop["values"][0]}"' if prop["values"] else ""
                        )
                elif prop["type"] == "LIST":
                    # Skip embeddings
                    if not prop.get("min_size") or prop["min_size"] > LIST_LIMIT:
                        continue
                    example = (
                        f'Min Size: {prop["min_size"]}, Max Size: {prop["max_size"]}'
                    )
                formatted_rel_props.append(
                    f"  - `{prop['property']}: {prop['type']}` {example}"
                )
    else:
        # Format node properties
        for label, props in schema["node_props"].items():
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in props]
            )
            formatted_node_props.append(f"{label} {{{props_str}}}")

        # Format relationship properties using structured_schema
        for type, props in schema["rel_props"].items():
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in props]
            )
            formatted_rel_props.append(f"{type} {{{props_str}}}")

    # Format relationships
    formatted_rels = [
        f"(:{el['start']})-[:{el['type']}]->(:{el['end']})"
        for el in schema["relationships"]
    ]

    return "\n".join(
        [
            "Node properties:",
            "\n".join(formatted_node_props),
            "Relationship properties:",
            "\n".join(formatted_rel_props),
            "The relationships:",
            "\n".join(formatted_rels),
        ]
    )


def _remove_backticks(text: str) -> str:
    return text.replace("`", "")


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
    refresh_schema (bool): A flag whether to refresh schema information
            at initialization. Default is True.
    enhanced_schema (bool): A flag whether to scan the database for
            example values and use them in the graph schema. Default is False.
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
        timeout: Optional[float] = None,
        sanitize: bool = False,
        refresh_schema: bool = True,
        *,
        driver_config: Optional[Dict] = None,
        enhanced_schema: bool = False,
    ) -> None:
        """Create a new Neo4j graph wrapper instance."""
        try:
            import neo4j
        except ImportError:
            raise ImportError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )

        url = get_from_dict_or_env({"url": url}, "url", "NEO4J_URI")
        # if username and password are "", assume Neo4j auth is disabled
        if username == "" and password == "":
            auth = None
        else:
            username = get_from_dict_or_env(
                {"username": username},
                "username",
                "NEO4J_USERNAME",
            )
            password = get_from_dict_or_env(
                {"password": password},
                "password",
                "NEO4J_PASSWORD",
            )
            auth = (username, password)
        database = get_from_dict_or_env(
            {"database": database}, "database", "NEO4J_DATABASE", "neo4j"
        )

        self._driver = neo4j.GraphDatabase.driver(
            url, auth=auth, **(driver_config or {})
        )
        self._database = database
        self.timeout = timeout
        self.sanitize = sanitize
        self._enhanced_schema = enhanced_schema
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
        if refresh_schema:
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

    def query(
        self,
        query: str,
        params: dict = {},
    ) -> List[Dict[str, Any]]:
        """Query Neo4j database.

        Args:
            query (str): The Cypher query to execute.
            params (dict): The parameters to pass to the query.

        Returns:
            List[Dict[str, Any]]: The list of dictionaries containing the query results.
        """
        from neo4j import Query
        from neo4j.exceptions import Neo4jError

        try:
            data, _, _ = self._driver.execute_query(
                Query(text=query, timeout=self.timeout),
                database_=self._database,
                parameters_=params,
            )
            json_data = [r.data() for r in data]
            if self.sanitize:
                json_data = [value_sanitize(el) for el in json_data]
            return json_data
        except Neo4jError as e:
            if not (
                (
                    (  # isCallInTransactionError
                        e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                        or e.code
                        == "Neo.DatabaseError.Transaction.TransactionStartFailed"
                    )
                    and "in an implicit transaction" in e.message  # type: ignore[operator]
                )
                or (  # isPeriodicCommitError
                    e.code == "Neo.ClientError.Statement.SemanticError"
                    and (
                        "in an open transaction is not possible" in e.message  # type: ignore[operator]
                        or "tried to execute in an explicit transaction" in e.message  # type: ignore[operator]
                    )
                )
            ):
                raise
        # fallback to allow implicit transactions
        with self._driver.session(database=self._database) as session:
            data = session.run(Query(text=query, timeout=self.timeout), params)  # type: ignore[assignment]
            json_data = [r.data() for r in data]
            if self.sanitize:
                json_data = [value_sanitize(el) for el in json_data]
            return json_data

    def refresh_schema(self) -> None:
        """
        Refreshes the Neo4j graph schema information.
        """
        from neo4j.exceptions import ClientError, CypherTypeError

        node_properties = [
            el["output"]
            for el in self.query(
                node_properties_query,
                params={"EXCLUDED_LABELS": EXCLUDED_LABELS + [BASE_ENTITY_LABEL]},
            )
        ]
        rel_properties = [
            el["output"]
            for el in self.query(
                rel_properties_query, params={"EXCLUDED_LABELS": EXCLUDED_RELS}
            )
        ]
        relationships = [
            el["output"]
            for el in self.query(
                rel_query,
                params={"EXCLUDED_LABELS": EXCLUDED_LABELS + [BASE_ENTITY_LABEL]},
            )
        ]

        # Get constraints & indexes
        try:
            constraint = self.query("SHOW CONSTRAINTS")
            index = self.query(
                "CALL apoc.schema.nodes() YIELD label, properties, type, size, "
                "valuesSelectivity WHERE type = 'RANGE' RETURN *, "
                "size * valuesSelectivity as distinctValues"
            )
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
        if self._enhanced_schema:
            schema_counts = self.query(
                "CALL apoc.meta.graphSample() YIELD nodes, relationships "
                "RETURN nodes, [rel in relationships | {name:apoc.any.property"
                "(rel, 'type'), count: apoc.any.property(rel, 'count')}]"
                " AS relationships"
            )
            # Update node info
            for node in schema_counts[0]["nodes"]:
                # Skip bloom labels
                if node["name"] in EXCLUDED_LABELS:
                    continue
                node_props = self.structured_schema["node_props"].get(node["name"])
                if not node_props:  # The node has no properties
                    continue
                enhanced_cypher = self._enhanced_schema_cypher(
                    node["name"], node_props, node["count"] < EXHAUSTIVE_SEARCH_LIMIT
                )
                # Due to schema-flexible nature of neo4j errors can happen
                try:
                    enhanced_info = self.query(enhanced_cypher)[0]["output"]
                    for prop in node_props:
                        if prop["property"] in enhanced_info:
                            prop.update(enhanced_info[prop["property"]])
                except CypherTypeError:
                    continue
            # Update rel info
            for rel in schema_counts[0]["relationships"]:
                # Skip bloom labels
                if rel["name"] in EXCLUDED_RELS:
                    continue
                rel_props = self.structured_schema["rel_props"].get(rel["name"])
                if not rel_props:  # The rel has no properties
                    continue
                enhanced_cypher = self._enhanced_schema_cypher(
                    rel["name"],
                    rel_props,
                    rel["count"] < EXHAUSTIVE_SEARCH_LIMIT,
                    is_relationship=True,
                )
                try:
                    enhanced_info = self.query(enhanced_cypher)[0]["output"]
                    for prop in rel_props:
                        if prop["property"] in enhanced_info:
                            prop.update(enhanced_info[prop["property"]])
                # Due to schema-flexible nature of neo4j errors can happen
                except CypherTypeError:
                    continue

        schema = _format_schema(self.structured_schema, self._enhanced_schema)

        self.schema = schema

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
        This is useful for tracing back the origin of data. Merges source
        documents based on the `id` property from the source document metadata
        if available; otherwise it calculates the MD5 hash of `page_content`
        for merging process. Defaults to False.
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
                        "constraint", []
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
            if not document.source.metadata.get("id"):
                document.source.metadata["id"] = md5(
                    document.source.page_content.encode("utf-8")
                ).hexdigest()

            # Remove backticks from node types
            for node in document.nodes:
                node.type = _remove_backticks(node.type)
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
                            "source_label": _remove_backticks(el.source.type),
                            "target": el.target.id,
                            "target_label": _remove_backticks(el.target.type),
                            "type": _remove_backticks(
                                el.type.replace(" ", "_").upper()
                            ),
                            "properties": el.properties,
                        }
                        for el in document.relationships
                    ]
                },
            )

    def _enhanced_schema_cypher(
        self,
        label_or_type: str,
        properties: List[Dict[str, Any]],
        exhaustive: bool,
        is_relationship: bool = False,
    ) -> str:
        if is_relationship:
            match_clause = f"MATCH ()-[n:`{label_or_type}`]->()"
        else:
            match_clause = f"MATCH (n:`{label_or_type}`)"

        with_clauses = []
        return_clauses = []
        output_dict = {}
        if exhaustive:
            for prop in properties:
                prop_name = prop["property"]
                prop_type = prop["type"]
                if prop_type == "STRING":
                    with_clauses.append(
                        (
                            f"collect(distinct substring(toString(n.`{prop_name}`)"
                            f", 0, 50)) AS `{prop_name}_values`"
                        )
                    )
                    return_clauses.append(
                        (
                            f"values:`{prop_name}_values`[..{DISTINCT_VALUE_LIMIT}],"
                            f" distinct_count: size(`{prop_name}_values`)"
                        )
                    )
                elif prop_type in [
                    "INTEGER",
                    "FLOAT",
                    "DATE",
                    "DATE_TIME",
                    "LOCAL_DATE_TIME",
                ]:
                    with_clauses.append(f"min(n.`{prop_name}`) AS `{prop_name}_min`")
                    with_clauses.append(f"max(n.`{prop_name}`) AS `{prop_name}_max`")
                    with_clauses.append(
                        f"count(distinct n.`{prop_name}`) AS `{prop_name}_distinct`"
                    )
                    return_clauses.append(
                        (
                            f"min: toString(`{prop_name}_min`), "
                            f"max: toString(`{prop_name}_max`), "
                            f"distinct_count: `{prop_name}_distinct`"
                        )
                    )
                elif prop_type == "LIST":
                    with_clauses.append(
                        (
                            f"min(size(n.`{prop_name}`)) AS `{prop_name}_size_min`, "
                            f"max(size(n.`{prop_name}`)) AS `{prop_name}_size_max`"
                        )
                    )
                    return_clauses.append(
                        f"min_size: `{prop_name}_size_min`, "
                        f"max_size: `{prop_name}_size_max`"
                    )
                elif prop_type in ["BOOLEAN", "POINT", "DURATION"]:
                    continue
                output_dict[prop_name] = "{" + return_clauses.pop() + "}"
        else:
            # Just sample 5 random nodes
            match_clause += " WITH n LIMIT 5"
            for prop in properties:
                prop_name = prop["property"]
                prop_type = prop["type"]

                # Check if indexed property, we can still do exhaustive
                prop_index = [
                    el
                    for el in self.structured_schema["metadata"]["index"]
                    if el["label"] == label_or_type
                    and el["properties"] == [prop_name]
                    and el["type"] == "RANGE"
                ]
                if prop_type == "STRING":
                    if (
                        prop_index
                        and prop_index[0].get("size") > 0
                        and prop_index[0].get("distinctValues") <= DISTINCT_VALUE_LIMIT
                    ):
                        distinct_values = self.query(
                            f"CALL apoc.schema.properties.distinct("
                            f"'{label_or_type}', '{prop_name}') YIELD value"
                        )[0]["value"]
                        return_clauses.append(
                            (
                                f"values: {distinct_values},"
                                f" distinct_count: {len(distinct_values)}"
                            )
                        )
                    else:
                        with_clauses.append(
                            (
                                f"collect(distinct substring(toString(n.`{prop_name}`)"
                                f", 0, 50)) AS `{prop_name}_values`"
                            )
                        )
                        return_clauses.append(f"values: `{prop_name}_values`")
                elif prop_type in [
                    "INTEGER",
                    "FLOAT",
                    "DATE",
                    "DATE_TIME",
                    "LOCAL_DATE_TIME",
                ]:
                    if not prop_index:
                        with_clauses.append(
                            f"collect(distinct toString(n.`{prop_name}`)) "
                            f"AS `{prop_name}_values`"
                        )
                        return_clauses.append(f"values: `{prop_name}_values`")
                    else:
                        with_clauses.append(
                            f"min(n.`{prop_name}`) AS `{prop_name}_min`"
                        )
                        with_clauses.append(
                            f"max(n.`{prop_name}`) AS `{prop_name}_max`"
                        )
                        with_clauses.append(
                            f"count(distinct n.`{prop_name}`) AS `{prop_name}_distinct`"
                        )
                        return_clauses.append(
                            (
                                f"min: toString(`{prop_name}_min`), "
                                f"max: toString(`{prop_name}_max`), "
                                f"distinct_count: `{prop_name}_distinct`"
                            )
                        )

                elif prop_type == "LIST":
                    with_clauses.append(
                        (
                            f"min(size(n.`{prop_name}`)) AS `{prop_name}_size_min`, "
                            f"max(size(n.`{prop_name}`)) AS `{prop_name}_size_max`"
                        )
                    )
                    return_clauses.append(
                        (
                            f"min_size: `{prop_name}_size_min`, "
                            f"max_size: `{prop_name}_size_max`"
                        )
                    )
                elif prop_type in ["BOOLEAN", "POINT", "DURATION"]:
                    continue

                output_dict[prop_name] = "{" + return_clauses.pop() + "}"

        with_clause = "WITH " + ",\n     ".join(with_clauses)
        return_clause = (
            "RETURN {"
            + ", ".join(f"`{k}`: {v}" for k, v in output_dict.items())
            + "} AS output"
        )

        # Combine all parts of the Cypher query
        cypher_query = "\n".join([match_clause, with_clause, return_clause])
        return cypher_query
