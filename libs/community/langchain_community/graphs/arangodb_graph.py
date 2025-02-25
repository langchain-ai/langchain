import itertools
import json
import os
import re
from collections import defaultdict
from importlib.util import find_spec
from math import ceil
from typing import Any, DefaultDict, Dict, List, Optional, Set

from langchain_core.embeddings import Embeddings

from langchain_community.graphs.graph_document import (
    Document,
    GraphDocument,
    Node,
    Relationship,
)
from langchain_community.graphs.graph_store import GraphStore

try:
    from arango.database import StandardDatabase
    from arango.graph import Graph

    ARANGO_INSTALLED = True
except ImportError:
    ARANGO_INSTALLED = False

SOURCE_VERTEX_COLLECTION = "SOURCE"
SOURCE_EDGE_COLLECTION = "HAS_SOURCE"
ENTITY_VERTEX_COLLECTION = "ENTITY"
ENTITY_EDGE_COLLECTION = "LINKS_TO"


class ArangoGraph(GraphStore):
    """ArangoDB wrapper for graph operations.

    Parameters:
    - db (arango.database.StandardDatabase): ArangoDB database instance.
    - generate_schema_on_init (bool): A flag whether to generate the schema
        on initialization. Default is True.
    - schema_sample_ratio (float): A ratio (0 to 1) to determine the
        ratio of documents/edges used (in relation to the Collection size)
        to render each Collection Schema. If 0, one document/edge
        is used per Collection. Default is 0.
    - schema_graph_name (str): The name of the graph to use to generate the schema. If
        None, the entire database will be used. Default is None.
    - schema_include_examples (bool): A flag whether to scan the database for
        example values and use them in the graph schema. Default is True.
    - schema_list_limit (int): The maximum number of elements to include in a list.
        If the list is longer than this limit, a string describing the list
        will be used in the schema instead. Default is 32.

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
        db: "StandardDatabase",
        generate_schema_on_init: bool = True,
        schema_sample_ratio: float = 0,
        schema_graph_name: Optional[str] = None,
        schema_include_examples: bool = True,
        schema_list_limit: int = 32,
    ) -> None:
        if not ARANGO_INSTALLED:
            m = "ArangoDB not installed, please install with pip install python-arango"
            raise ImportError(m)

        self.__db = db
        self.__async_db = db.begin_async_execution()

        self.__schema = {}
        if generate_schema_on_init:
            self.__schema = self.generate_schema(
                schema_sample_ratio,
                schema_graph_name,
                schema_include_examples,
                schema_list_limit,
            )

    @property
    def db(self) -> "StandardDatabase":
        return self.__db

    @property
    def schema(self) -> Dict[str, Any]:
        """Returns the schema of the Graph Database as a structured object"""
        return self.__schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the schema of the Graph Database as a structured object"""
        return self.__schema

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph Database as a string"""
        return json.dumps(self.__schema)

    def set_schema(self, schema: Dict[str, Any]) -> None:
        """Sets a custom schema for the ArangoDB Database."""
        self.__schema = schema

    def refresh_schema(
        self,
        sample_ratio: float = 0,
        graph_name: Optional[str] = None,
        include_examples: bool = True,
        list_limit: int = 32,
    ) -> None:
        """
        Refresh the graph schema information.

        Parameters:
        - sample_ratio (float): A ratio (0 to 1) to determine the
        ratio of documents/edges used (in relation to the Collection size)
        to render each Collection Schema. If 0, one document/edge
        is used per Collection.
        - graph_name (str): The name of the graph to use to generate the schema. If
            None, the entire database will be used.
        - include_examples (bool): A flag whether to scan the database for
            example values and use them in the graph schema. Default is True.
        - list_limit (int): The maximum number of elements to include in a list.
            If the list is longer than this limit, a string describing the list
            will be used in the schema instead. Default is 32.
        """
        self.__schema = self.generate_schema(
            sample_ratio, graph_name, include_examples, list_limit
        )

    def generate_schema(
        self,
        sample_ratio: float = 0,
        graph_name: Optional[str] = None,
        include_examples: bool = True,
        list_limit: int = 32,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates the schema of the ArangoDB Database and returns it

        Parameters:
        - sample_ratio (float): A ratio (0 to 1) to determine the
        ratio of documents/edges used (in relation to the Collection size)
        to render each Collection Schema. If 0, one document/edge
        is used per Collection.
        - graph_name (str): The name of the graph to use to generate the schema. If
            None, the entire database will be used.
        - include_examples (bool): A flag whether to scan the database for
            example values and use them in the graph schema. Default is True.
        - list_limit (int): The maximum number of elements to include in a list.
            If the list is longer than this limit, a string describing the list
            will be used in the schema instead. Default is 32.
        """
        if not 0 <= sample_ratio <= 1:
            raise ValueError("**sample_ratio** value must be in between 0 to 1")

        graph_schema: List[Dict[str, Any]] = []
        if graph_name:
            # Fetch a single graph
            graph: Graph = self.db.graph(graph_name)
            edge_definitions = graph.edge_definitions()

            graph_schema = [{"name": graph_name, "edge_definitions": edge_definitions}]

            # Fetch graph-specific collections
            collection_names = set(graph.vertex_collections())
            for edge_definition in edge_definitions:
                collection_names.add(edge_definition["edge_collection"])

        else:
            # Fetch all graphs
            graph_schema = [
                {"graph_name": g["name"], "edge_definitions": g["edge_definitions"]}
                for g in self.db.graphs()
            ]

            # Fetch all collections
            collection_names = {
                collection["name"] for collection in self.db.collections()
            }

        # Stores the schema of every ArangoDB Document/Edge collection
        collection_schema: List[Dict[str, Any]] = []
        for collection in self.db.collections():
            if collection["system"] or collection["name"] not in collection_names:
                continue

            # Extract collection name, type, and size
            col_name: str = collection["name"]
            col_type: str = collection["type"]
            col_size: int = self.db.collection(col_name).count()

            # Set number of ArangoDB documents/edges to retrieve
            limit_amount = ceil(sample_ratio * col_size) or 1

            aql = f"""
                FOR doc in @@col_name
                    LIMIT {limit_amount}
                    RETURN doc
            """

            doc: Dict[str, Any]
            properties: List[Dict[str, str]] = []
            for doc in self.db.aql.execute(aql, bind_vars={"@col_name": col_name}):
                for key, value in doc.items():
                    properties.append({"name": key, "type": type(value).__name__})

            collection_schema_entry = {
                "name": col_name,
                "type": col_type,
                "size": col_size,
                "properties": properties,
            }

            if include_examples and col_size > 0:
                collection_schema_entry["example"] = self._sanitize_input(
                    doc, list_limit
                )

            collection_schema.append(collection_schema_entry)

        return {"graph_schema": graph_schema, "collection_schema": collection_schema}

    def query(
        self,
        query: str,
        params: dict = {},
    ) -> List[Dict[str, Any]]:
        """
        Execute an AQL query and return the results.

        Parameters:
        - query (str): The AQL query to execute.
        - top_k (int): The number of results to return. If None, all results
            are returned.
        - kwargs: Additional keyword arguments to pass to the AQL query.

        Returns:
        - A list of dictionaries containing the query results.
        """
        top_k = params.pop("top_k", None)
        list_limit = params.pop("list_limit", 32)
        cursor = self.__db.aql.execute(query, **params)
        return [
            self._sanitize_input(doc, list_limit)
            for doc in itertools.islice(cursor, top_k)
        ]

    def explain(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """
        Explain an AQL query without executing it.

        Parameters:
        - query (str): The AQL query to explain.
        - args: Additional positional arguments to pass to the AQL query.
        - kwargs: Additional keyword arguments to pass to the AQL query.

        Returns:
        - A list of dictionaries containing the query explanation.
        """
        return self.__db.aql.explain(query)

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        graph_name: Optional[str] = None,
        update_graph_definition_if_exists: bool = False,
        batch_size: int = 1000,
        use_one_entity_collection: bool = True,
        insert_async: bool = False,
        source_collection_name: str | None = None,
        source_edge_collection_name: str | None = None,
        entity_collection_name: str | None = None,
        entity_edge_collection_name: str | None = None,
        embeddings: Embeddings | None = None,
        embedding_field: str = "embedding",
        embed_source: bool = False,
        embed_nodes: bool = False,
        embed_relationships: bool = False,
    ) -> None:
        """
        Constructs nodes & relationships in the graph based on the
        provided GraphDocument objects.

        Parameters:
        - graph_documents (List[GraphDocument]): A list of GraphDocument objects
        that contain the nodes and relationships to be added to the graph. Each
        GraphDocument should encapsulate the structure of part of the graph,
        including nodes, relationships, and the source document information.
        - include_source (bool, optional): If True, stores the source document
        and links it to nodes in the graph using the HAS_SOURCE relationship.
        This is useful for tracing back the origin of data. Merges source
        documents based on the `id` property from the source document if available,
        otherwise it calculates the Farmhash hash of `page_content`
        for merging process. Defaults to False.
        - graph_name (str): The name of the ArangoDB General Graph to create. If None,
            no graph will be created.
        - update_graph_definition_if_exists (bool): If True, updates the graph
            Edge Definitions
        if it already exists. Defaults to False. Not used if `graph_name` is None. It is
        recommended to set this to True if `use_one_entity_collection` is set to False.
        - batch_size (int): The number of nodes/edges to insert in a single batch.
        - use_one_entity_collection (bool): If True, all nodes are stored in a single
        entity collection. If False, nodes are stored in separate collections based
        on their type. Defaults to True.
        - insert_async (bool): If True, inserts data asynchronously. Defaults to False.
        - source_collection_name (str): The name of the collection to store the source
        documents. Defaults to "SOURCE".
        - source_edge_collection_name (str): The name of the edge collection to store
        the relationships between source documents and nodes. Defaults to "HAS_SOURCE".
        - entity_collection_name (str): The name of the collection to store the nodes.
        Defaults to "ENTITY". Only used if `use_one_entity_collection` is True.
        - entity_edge_collection_name (str): The name of the edge collection to store
        the relationships between nodes. Defaults to "LINKS_TO". Only used if
        `use_one_entity_collection` is True.
        - embeddings (Embeddings): An Embeddings object to use for embedding the source,
        nodes and relationships. Defaults to None.
        - embedding_field (set[str]): The field name to store the embedding. Defaults
            to "embedding". Only used if `embedding` is not None, and `embed_source`,
            `embed_nodes`, or `embed_relationships` is True.
        - embed_source (bool): If True, embeds the source document. Defaults to False.
        - embed_nodes (bool): If True, embeds the nodes. Defaults to False.
        - embed_relationships (bool): If True, embeds the relationships.
            Defaults to False.
        """
        if not graph_documents:
            return

        if find_spec("farmhash") is None:
            m = "Farmhash not install=, please install with `pip install cityhash`."
            raise ImportError(m)

        if not embeddings and (embed_source or embed_nodes or embed_relationships):
            m = "**embedding** is required to embed source, nodes, or relationships."
            raise ValueError(m)

        def embed_text(text: str) -> list[float]:
            if not embeddings:
                raise ValueError("**embedding** is required to embed text.")

            res: Any = embeddings.embed_documents([text])[0]

            while type(res) is list:
                if type(res[0]) is float:
                    break

                res = res[0]

            return res

        #########
        # Setup #
        #########

        suffix = f"{graph_name}_" if graph_name else ""
        if not source_collection_name:
            source_collection_name = f"{suffix}{SOURCE_VERTEX_COLLECTION}"
        if not source_edge_collection_name:
            source_edge_collection_name = f"{suffix}{SOURCE_EDGE_COLLECTION}"
        if not entity_collection_name:
            entity_collection_name = f"{suffix}{ENTITY_VERTEX_COLLECTION}"
        if not entity_edge_collection_name:
            entity_edge_collection_name = f"{suffix}{ENTITY_EDGE_COLLECTION}"

        insertion_db = self.__async_db if insert_async else self.__db
        nodes: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)
        edges: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)
        edge_definitions_dict: DefaultDict[str, DefaultDict[str, Set[str]]] = (
            defaultdict(lambda: defaultdict(set))
        )

        if include_source:
            self._create_collection(source_collection_name)
            self._create_collection(source_edge_collection_name, is_edge=True)

            from_cols = {entity_collection_name} if use_one_entity_collection else set()

            edge_definitions_dict[source_edge_collection_name][
                "from_vertex_collections"
            ] = from_cols
            edge_definitions_dict[source_edge_collection_name][
                "to_vertex_collections"
            ] = {source_collection_name}

        if use_one_entity_collection:
            self._create_collection(entity_collection_name)
            self._create_collection(entity_edge_collection_name, is_edge=True)

            edge_definitions_dict[entity_edge_collection_name][
                "from_vertex_collections"
            ] = {entity_collection_name}
            edge_definitions_dict[entity_edge_collection_name][
                "to_vertex_collections"
            ] = {entity_collection_name}

        process_node_fn = (
            self._process_node_as_entity
            if use_one_entity_collection
            else self._process_node_as_type
        )

        process_edge_fn = (
            self._process_edge_as_entity
            if use_one_entity_collection
            else self._process_edge_as_type
        )

        #######################
        # Document Processing #
        #######################

        for document in graph_documents:
            source_id_hash = None

            # 1. Process Source Document
            if include_source:
                source_embedding = (
                    embed_text(document.source.page_content) if embed_source else None
                )

                source_id_hash = self._process_source(
                    document.source,
                    source_collection_name,
                    source_embedding,
                    embedding_field,
                )

            # 2. Process Nodes
            node_key_map = {}
            for i, node in enumerate(document.nodes, 1):
                node.id = str(node.id)
                node_key = self._hash(node.id)
                node_key_map[node.id] = node_key

                if embed_nodes:
                    node.properties[embedding_field] = embed_text(node.id)

                node_type = process_node_fn(
                    node_key, node, nodes, entity_collection_name
                )

                # 2.1 Link Source Document to Node
                if include_source:
                    edges[source_edge_collection_name].append(
                        {
                            "_from": f"{node_type}/{node_key}",
                            "_to": f"{source_collection_name}/{source_id_hash}",
                        }
                    )

                    if not use_one_entity_collection:
                        edge_definitions_dict[source_edge_collection_name][
                            "from_vertex_collections"
                        ].add(node_type)

                # 2.2 Batch Insert
                if i % batch_size == 0:
                    self._import_data(insertion_db, nodes, is_edge=False)
                    self._import_data(insertion_db, edges, is_edge=True)

            self._import_data(insertion_db, nodes, is_edge=False)
            self._import_data(insertion_db, edges, is_edge=True)

            # 3. Process Edges
            for i, edge in enumerate(document.relationships, 1):
                source, target = edge.source, edge.target

                edge_str = f"{source.id} {edge.type} {target.id}"

                if embed_relationships:
                    edge.properties[embedding_field] = embed_text(edge_str)

                if include_source:
                    edge.properties["source_id"] = source_id_hash

                source_key = self._get_node_key(
                    source,
                    nodes,
                    node_key_map,
                    entity_collection_name,
                    process_node_fn,
                )

                target_key = self._get_node_key(
                    target,
                    nodes,
                    node_key_map,
                    entity_collection_name,
                    process_node_fn,
                )

                edge_key = self._hash(edge_str)
                process_edge_fn(
                    edge,
                    edge_key,
                    source_key,
                    target_key,
                    edges,
                    entity_collection_name,
                    entity_edge_collection_name,
                    edge_definitions_dict,
                )

                # 3.1 Batch Insert
                if i % batch_size == 0:
                    self._import_data(insertion_db, edges, is_edge=True)
                    self._import_data(insertion_db, nodes, is_edge=False)

            self._import_data(insertion_db, edges, is_edge=True)
            self._import_data(insertion_db, nodes, is_edge=False)

        ##################
        # Graph Creation #
        ##################

        if graph_name:
            edge_definitions = [
                {
                    "edge_collection": k,
                    "from_vertex_collections": list(v["from_vertex_collections"]),
                    "to_vertex_collections": list(v["to_vertex_collections"]),
                }
                for k, v in edge_definitions_dict.items()
            ]

            if not self.db.has_graph(graph_name):
                self.db.create_graph(graph_name, edge_definitions)

            elif update_graph_definition_if_exists:
                graph = self.db.graph(graph_name)

                for e_d in edge_definitions:
                    if not graph.has_edge_definition(e_d["edge_collection"]):
                        graph.create_edge_definition(*e_d.values())
                    else:
                        graph.replace_edge_definition(*e_d.values())

        # Refresh schema after insertions
        self.refresh_schema()

    @classmethod
    def from_db_credentials(
        cls,
        url: Optional[str] = None,
        dbname: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Any:
        """Convenience constructor that builds Arango DB from credentials.

        Args:
            url: Arango DB url. Can be passed in as named arg or set as environment
                var ``ARANGODB_URL``. Defaults to "http://localhost:8529".
            dbname: Arango DB name. Can be passed in as named arg or set as
                environment var ``ARANGODB_DBNAME``. Defaults to "_system".
            username: Can be passed in as named arg or set as environment var
                ``ARANGODB_USERNAME``. Defaults to "root".
            password: Can be passed ni as named arg or set as environment var
                ``ARANGODB_PASSWORD``. Defaults to "".

        Returns:
            An arango.database.StandardDatabase.
        """
        db = get_arangodb_client(
            url=url, dbname=dbname, username=username, password=password
        )
        return cls(db)

    def _import_data(
        self,
        db: "StandardDatabase",
        data: Dict[str, List[Dict[str, Any]]],
        is_edge: bool,
    ) -> None:
        """Imports data into the ArangoDB database in bulk."""
        for collection, batch in data.items():
            self._create_collection(collection, is_edge)
            db.collection(collection).import_bulk(batch, on_duplicate="update")

        data.clear()

    def _create_collection(
        self, collection_name: str, is_edge: bool = False, **kwargs: Any
    ) -> None:
        """Creates a collection in the ArangoDB database if it does not exist."""
        if not self.db.has_collection(collection_name):
            self.db.create_collection(collection_name, edge=is_edge, **kwargs)

    def _process_node_as_entity(
        self,
        node_key: str,
        node: Node,
        nodes: DefaultDict[str, list],
        entity_collection_name: str,
    ) -> str:
        """Processes a Graph Document Node into ArangoDB as a unanimous Entity."""
        nodes[entity_collection_name].append(
            {
                **node.properties,
                "_key": node_key,
                "name": node.id,
                "type": node.type,
            }
        )
        return entity_collection_name

    def _process_node_as_type(
        self, node_key: str, node: Node, nodes: DefaultDict[str, list], _: str
    ) -> str:
        """Processes a Graph Document Node into ArangoDB based on its Node Type."""
        node_type = self._sanitize_collection_name(node.type)
        nodes[node_type].append({"_key": node_key, "name": node.id, **node.properties})
        return node_type

    def _process_edge_as_entity(
        self,
        edge: Relationship,
        edge_key: str,
        source_key: str,
        target_key: str,
        edges: DefaultDict[str, list],
        entity_collection_name: str,
        entity_edge_collection_name: str,
        _: DefaultDict[str, DefaultDict[str, set[str]]],
    ) -> None:
        """Processes a Graph Document Edge into ArangoDB as a unanimous Entity."""
        edges[entity_edge_collection_name].append(
            {
                **edge.properties,
                "_key": edge_key,
                "_from": f"{entity_collection_name}/{source_key}",
                "_to": f"{entity_collection_name}/{target_key}",
                "type": edge.type,
            }
        )

    def _process_edge_as_type(
        self,
        edge: Relationship,
        edge_key: str,
        source_key: str,
        target_key: str,
        edges: DefaultDict[str, list],
        _1: str,
        _2: str,
        edge_definitions_dict: DefaultDict[str, DefaultDict[str, set[str]]],
    ) -> None:
        """Processes a Graph Document Edge into ArangoDB based on its Edge Type."""
        source: Node = edge.source
        target: Node = edge.target

        edge_type = self._sanitize_collection_name(edge.type)
        source_type = self._sanitize_collection_name(source.type)
        target_type = self._sanitize_collection_name(target.type)

        edge_definitions_dict[edge_type]["from_vertex_collections"].add(source_type)
        edge_definitions_dict[edge_type]["to_vertex_collections"].add(target_type)

        edges[edge_type].append(
            {
                **edge.properties,
                "_key": edge_key,
                "_from": f"{source_type}/{source_key}",
                "_to": f"{target_type}/{target_key}",
            }
        )

    def _get_node_key(
        self,
        node: Node,
        nodes: DefaultDict[str, list],
        node_key_map: Dict[str, str],
        entity_collection_name: str,
        process_node_fn: Any,
    ) -> str:
        """Gets the key of a node and processes it if it doesn't exist."""
        node.id = str(node.id)
        if node.id in node_key_map:
            return node_key_map[node.id]

        node_key = self._hash(node.id)
        node_key_map[node.id] = node_key
        process_node_fn(node_key, node, nodes, entity_collection_name)

        return node_key

    def _process_source(
        self,
        source: Document,
        source_collection_name: str,
        source_embedding: list[float] | None,
        embedding_field: str,
    ) -> str:
        """Processes a Graph Document Source into ArangoDB."""
        source_id = self._hash(
            source.id if source.id else source.page_content.encode("utf-8")
        )

        doc = {
            **source.metadata,
            "_key": source_id,
            "text": source.page_content,
            "type": source.type,
        }

        if source_embedding:
            doc[embedding_field] = source_embedding

        self.db.collection(source_collection_name).insert(doc, overwrite=True)

        return source_id

    def _hash(self, value: Any) -> str:
        """Applies the Farmhash hash function to a value."""
        import farmhash

        try:
            value_str = str(value)
        except Exception:
            raise ValueError("Value must be a string or have a string representation.")

        return str(farmhash.Fingerprint64(value_str))

    def _sanitize_collection_name(self, name: str) -> str:
        """
        Modifies a string to adhere to ArangoDB collection name rules.

        - Trims the name to 256 characters if it's too long.
        - Replaces invalid characters with underscores (_).
        - Ensures the name starts with a letter (prepends 'a' if needed).
        """
        if not name:
            raise ValueError("Collection name cannot be empty.")

        name = name[:256]

        # Replace invalid characters with underscores
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        # Ensure the name starts with a letter; prepend 'a' if not
        if not re.match(r"^[a-zA-Z]", name):
            name = f"Collection_{name}"

        return name

    def _sanitize_input(self, d: Any, list_limit: int) -> Any:
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
                    sanitized_value = self._sanitize_input(value, list_limit)
                    if sanitized_value is not None:
                        new_dict[key] = sanitized_value
                elif isinstance(value, list):
                    if len(value) < list_limit:
                        sanitized_value = self._sanitize_input(value, list_limit)
                        if sanitized_value is not None:
                            new_dict[key] = sanitized_value
                else:
                    new_dict[key] = value
            return new_dict
        elif isinstance(d, list):
            if len(d) == 0:
                return d
            elif len(d) < list_limit:
                arr = []
                for item in d:
                    sanitized_item = self._sanitize_input(item, list_limit)
                    if sanitized_item is not None:
                        arr.append(sanitized_item)
                return arr
            else:
                return f"List of {len(d)} elements of type {type(d[0])}"
        else:
            return d


def get_arangodb_client(
    url: Optional[str] = None,
    dbname: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Any:
    """Get the Arango DB client from credentials.

    Args:
        url: Arango DB url. Can be passed in as named arg or set as environment
            var ``ARANGODB_URL``. Defaults to "http://localhost:8529".
        dbname: Arango DB name. Can be passed in as named arg or set as
            environment var ``ARANGODB_DBNAME``. Defaults to "_system".
        username: Can be passed in as named arg or set as environment var
            ``ARANGODB_USERNAME``. Defaults to "root".
        password: Can be passed ni as named arg or set as environment var
            ``ARANGODB_PASSWORD``. Defaults to "".

    Returns:
        An arango.database.StandardDatabase.
    """
    try:
        from arango import ArangoClient
    except ImportError as e:
        m = "Unable to import arango, please install with `pip install python-arango`."
        raise ImportError(m) from e

    _url: str = url or os.environ.get("ARANGODB_URL", "http://localhost:8529")  # type: ignore[assignment]
    _dbname: str = dbname or os.environ.get("ARANGODB_DBNAME", "_system")  # type: ignore[assignment]
    _username: str = username or os.environ.get("ARANGODB_USERNAME", "root")  # type: ignore[assignment]
    _password: str = password or os.environ.get("ARANGODB_PASSWORD", "")  # type: ignore[assignment]

    return ArangoClient(_url).db(_dbname, _username, _password, verify=True)
