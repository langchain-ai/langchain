from hashlib import md5
from typing import Any, Dict, List, Tuple

from langchain_community.graphs.graph_document import GraphDocument, Relationship


class KuzuGraph:
    """Kùzu wrapper for graph operations.

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
        self, db: Any, database: str = "kuzu", allow_dangerous_requests: bool = False
    ) -> None:
        """Initializes the Kùzu graph database connection."""

        if allow_dangerous_requests is not True:
            raise ValueError(
                "The KuzuGraph class is a powerful tool that can be used to execute "
                "arbitrary queries on the database. To enable this functionality, "
                "set the `allow_dangerous_requests` parameter to `True` when "
                "constructing the KuzuGraph object."
            )

        try:
            import kuzu
        except ImportError:
            raise ImportError(
                "Could not import Kùzu python package."
                "Please install Kùzu with `pip install kuzu`."
            )
        self.db = db
        self.conn = kuzu.Connection(self.db)
        self.database = database
        self.refresh_schema()

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Kùzu database"""
        return self.schema

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Kùzu database"""
        result = self.conn.execute(query, params)
        column_names = result.get_column_names()
        return_list = []
        while result.has_next():
            row = result.get_next()
            return_list.append(dict(zip(column_names, row)))
        return return_list

    def refresh_schema(self) -> None:
        """Refreshes the Kùzu graph schema information"""
        node_properties = []
        node_table_names = self.conn._get_node_table_names()
        for table_name in node_table_names:
            current_table_schema = {"properties": [], "label": table_name}
            properties = self.conn._get_node_property_names(table_name)
            for property_name in properties:
                property_type = properties[property_name]["type"]
                list_type_flag = ""
                if properties[property_name]["dimension"] > 0:
                    if "shape" in properties[property_name]:
                        for s in properties[property_name]["shape"]:
                            list_type_flag += f"[{s}]"
                    else:
                        for i in range(properties[property_name]["dimension"]):
                            list_type_flag += "[]"
                property_type += list_type_flag
                current_table_schema["properties"].append(
                    (property_name, property_type)
                )
            node_properties.append(current_table_schema)

        relationships = []
        rel_tables = self.conn._get_rel_table_names()
        for table in rel_tables:
            relationships.append(
                f"(:{table['src']})-[:{table['name']}]->(:{table['dst']})"
            )

        rel_properties = []
        for table in rel_tables:
            table_name = table["name"]
            current_table_schema = {"properties": [], "label": table_name}
            query_result = self.conn.execute(
                f"CALL table_info('{table_name}') RETURN *;"
            )
            while query_result.has_next():
                row = query_result.get_next()
                prop_name = row[1]
                prop_type = row[2]
                current_table_schema["properties"].append((prop_name, prop_type))
            rel_properties.append(current_table_schema)

        self.schema = (
            f"Node properties: {node_properties}\n"
            f"Relationships properties: {rel_properties}\n"
            f"Relationships: {relationships}\n"
        )

    def _create_chunk_node_table(self) -> None:
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Chunk (
                id STRING,
                text STRING,
                type STRING,
                PRIMARY KEY(id)
            );
            """
        )

    def _create_entity_node_table(self, node_label: str) -> None:
        self.conn.execute(
            f"""
            CREATE NODE TABLE IF NOT EXISTS {node_label} (
                id STRING,
                type STRING,
                PRIMARY KEY(id)
            );
            """
        )

    def _create_entity_relationship_table(self, rel: Relationship) -> None:
        self.conn.execute(
            f"""
            CREATE REL TABLE IF NOT EXISTS {rel.type} (
                FROM {rel.source.type} TO {rel.target.type}
            );
            """
        )

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        allowed_relationships: List[Tuple[str, str, str]],
        include_source: bool = False,
    ) -> None:
        """
        Adds a list of `GraphDocument` objects that represent nodes and relationships
        in a graph to a Kùzu backend.

        Parameters:
          - graph_documents (List[GraphDocument]): A list of `GraphDocument` objects
            that contain the nodes and relationships to be added to the graph. Each
            `GraphDocument` should encapsulate the structure of part of the graph,
            including nodes, relationships, and the source document information.

          - allowed_relationships (List[Tuple[str, str, str]]): A list of allowed
            relationships that exist in the graph. Each tuple contains three elements:
            the source node type, the relationship type, and the target node type.
            Required for Kùzu, as the names of the relationship tables that need to
            pre-exist are derived from these tuples.

          - include_source (bool): If True, stores the source document
            and links it to nodes in the graph using the `MENTIONS` relationship.
            This is useful for tracing back the origin of data. Merges source
            documents based on the `id` property from the source document metadata
            if available; otherwise it calculates the MD5 hash of `page_content`
            for merging process. Defaults to False.
        """
        # Get unique node labels in the graph documents
        node_labels = list(
            {node.type for document in graph_documents for node in document.nodes}
        )

        for document in graph_documents:
            # Add chunk nodes and create source document relationships if include_source
            # is True
            if include_source:
                self._create_chunk_node_table()
                if not document.source.metadata.get("id"):
                    # Add a unique id to each document chunk via an md5 hash
                    document.source.metadata["id"] = md5(
                        document.source.page_content.encode("utf-8")
                    ).hexdigest()

                self.conn.execute(
                    f"""
                    MERGE (c:Chunk {{id: $id}}) 
                        SET c.text = $text,
                            c.type = "text_chunk"
                    """,  # noqa: F541
                    parameters={
                        "id": document.source.metadata["id"],
                        "text": document.source.page_content,
                    },
                )

            for node_label in node_labels:
                self._create_entity_node_table(node_label)

            # Add entity nodes from data
            for node in document.nodes:
                self.conn.execute(
                    f"""
                    MERGE (e:{node.type} {{id: $id}})
                        SET e.type = "entity"
                    """,
                    parameters={"id": node.id},
                )
                if include_source:
                    # If include_source is True, we need to create a relationship table
                    # between the chunk nodes and the entity nodes
                    self._create_chunk_node_table()
                    ddl = "CREATE REL TABLE GROUP IF NOT EXISTS MENTIONS ("
                    table_names = []
                    for node_label in node_labels:
                        table_names.append(f"FROM Chunk TO {node_label}")
                    table_names = list(set(table_names))
                    ddl += ", ".join(table_names)
                    # Add common properties for all the tables here
                    ddl += ", label STRING, triplet_source_id STRING)"
                    if ddl:
                        self.conn.execute(ddl)

                    # Only allow relationships that exist in the schema
                    if node.type in node_labels:
                        self.conn.execute(
                            f"""
                            MATCH (c:Chunk {{id: $id}}),
                                  (e:{node.type} {{id: $node_id}})
                            MERGE (c)-[m:MENTIONS]->(e)
                          SET m.triplet_source_id = $id
                            """,
                            parameters={
                                "id": document.source.metadata["id"],
                                "node_id": node.id,
                            },
                        )

            # Add entity relationships
            for rel in document.relationships:
                self._create_entity_relationship_table(rel)
                # Create relationship
                source_label = rel.source.type
                source_id = rel.source.id
                target_label = rel.target.type
                target_id = rel.target.id
                self.conn.execute(
                    f"""
                    MATCH (e1:{source_label} {{id: $source_id}}),
                            (e2:{target_label} {{id: $target_id}})
                    MERGE (e1)-[:{rel.type}]->(e2)
                    """,
                    parameters={
                        "source_id": source_id,
                        "target_id": target_id,
                    },
                )
