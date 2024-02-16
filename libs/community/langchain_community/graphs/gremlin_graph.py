import hashlib
import sys
from typing import Any, Dict, List, Optional

from langchain_core.utils import get_from_env

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore


class GremlinGraph(GraphStore):
    """Gremlin wrapper for graph operations.
    Parameters:
    url (Optional[str]): The URL of the Gremlin database server or env GREMLIN_URI
    username (Optional[str]): The collection-identifier like '/dbs/database/colls/graph'
                               or env GREMLIN_USERNAME if none provided
    password (Optional[str]): The connection-key for database authentication
                              or env GREMLIN_PASSWORD if none provided
    traversal_source (str): The traversal source to use for queries. Defaults to 'g'.
    message_serializer (Optional[Any]): The message serializer to use for requests.
                                        Defaults to serializer.GraphSONSerializersV2d0()
    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.

    *Implementation detais*:
        The Gremlin queries are designed to work with Azure CosmosDB limitations
    """

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        pass

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        traversal_source: str = "g",
        message_serializer: Optional[Any] = None,
    ) -> None:
        """Create a new Gremlin graph wrapper instance."""
        try:
            import asyncio

            from gremlin_python.driver import client, serializer

            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except ImportError:
            raise ValueError(
                "Please install gremlin-python first: " "`pip3 install gremlinpython"
            )

        self.client = client.Client(
            url=get_from_env("url", "GREMLIN_URI", url),
            traversal_source=traversal_source,
            username=get_from_env("username", "GREMLIN_USERNAME", username),
            password=get_from_env("password", "GREMLIN_PASSWORD", password),
            message_serializer=message_serializer
            if message_serializer
            else serializer.GraphSONSerializersV2d0(),
        )

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Gremlin database"""
        if self.schema is None:
            self.refresh_schema()
        return self.schema

    def refresh_schema(self) -> None:
        """
        Refreshes the Gremlin graph schema information.
        """
        vertex_schema = self.client.submit("g.V().label().dedup()").all().result()
        edge_schema = self.client.submit("g.E().label().dedup()").all().result()

        self.schema = "\n".join(
            [
                "Node labes are the following:",
                ",".join(vertex_schema),
                "Edge labes are the following:",
                ",".join(edge_schema),
            ]
        )

    def query(self, query: str) -> List[Dict[str, Any]]:
        q = self.client.submit(query)
        return q.all().result()

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """
        Take GraphDocument as input as uses it to construct a graph.
        """
        node_cache = {}
        for document in graph_documents:
            if include_source:
                # Create document vertex
                doc_props = {
                    "page_content": document.source.page_content,
                    "metadata": document.source.metadata,
                }
                doc_id = hashlib.md5(document.source.page_content.encode()).hexdigest()
                doc_node = self.add_node("Document", doc_id, doc_props, node_cache)

            # Import nodes to vertexes
            for el in document.nodes:
                node = self.add_node(el.type, el.id, el.properties)
                if include_source:
                    # Add Edge to document for each node
                    self.add_edge("source", doc_node, node, {})
                    self.add_edge("document", node, doc_node, {})

            # Edges
            for el in document.relationships:
                # Find or create the source vertex
                source = self.add_node(
                    el.source.type, el.source.id, el.source.properties, node_cache
                )
                # Find or create the target vertex
                target = self.add_node(
                    el.target.type, el.target.id, el.target.properties, node_cache
                )
                # Find or create the edge
                self.add_edge(el.type, source, target, el.properties)

    def build_vertex_query(self, label_value: str, id_value, properties: dict):
        base_query = (
            f"g.V().hasLabel('{label_value}').has('id','{id_value}').fold()"
            + f".coalesce(unfold(),addV('{label_value}')"
            + f".property('id','{id_value}').property('type', '{label_value}')"
        )
        for key, value in properties.items():
            base_query += f".property('{key}', '{value}')"

        return base_query + ")"

    def build_edge_query(
        self, type: str, source_node: dict, target_node: dict, properties: dict
    ):
        source_query = (
            f".hasLabel('{source_node['label']}')"
            + f".has('type', '{source_node['label']}')"
            + f".has('id','{source_node['id']}')"
        )
        target_query = (
            f".hasLabel('{target_node['label']}')"
            + f".has('type', '{target_node['label']}').has('id','{target_node['id']}')"
        )

        base_query = f""""g.V(){source_query}.as('a')  
            .V(){target_query}.as('b') 
            .choose(
                __.inE('{type}').where(outV().as('a')),
                __.identity(),
                __.addE('{type}').from('a').to('b')
            )        
            """.replace("\n", "").replace("\t", "")
        for key, value in properties.items():
            base_query += f".property('{key}', '{value}')"

        return base_query

    def add_node(self, type: str, id: str, properties: dict, node_cache: dict = {}):
        if id in node_cache:
            return node_cache[id]
        else:
            query = self.build_vertex_query(type, id, properties)
            node = self.client.submit(query).all().result()[0]
            node_cache[id] = node
            return node

    def add_edge(self, type: str, source: dict, target: dict, properties: dict):
        query = self.build_edge_query(type, source, target, properties)
        return self.client.submit(query).all().result()[0]
