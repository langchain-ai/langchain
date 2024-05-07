import hashlib
import sys
from typing import Any, Dict, List, Optional, Union

from langchain_core.utils import get_from_env

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
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

    *Implementation details*:
        The Gremlin queries are designed to work with Azure CosmosDB limitations
    """

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        return self.structured_schema

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
            raise ImportError(
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
        self.schema: str = ""

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Gremlin database"""
        if len(self.schema) == 0:
            self.refresh_schema()
        return self.schema

    def refresh_schema(self) -> None:
        """
        Refreshes the Gremlin graph schema information.
        """
        vertex_schema = self.client.submit("g.V().label().dedup()").all().result()
        edge_schema = self.client.submit("g.E().label().dedup()").all().result()
        vertex_properties = (
            self.client.submit(
                "g.V().group().by(label).by(properties().label().dedup().fold())"
            )
            .all()
            .result()[0]
        )

        self.structured_schema = {
            "vertex_labels": vertex_schema,
            "edge_labels": edge_schema,
            "vertice_props": vertex_properties,
        }

        self.schema = "\n".join(
            [
                "Vertex labels are the following:",
                ",".join(vertex_schema),
                "Edge labes are the following:",
                ",".join(edge_schema),
                f"Vertices have following properties:\n{vertex_properties}",
            ]
        )

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        q = self.client.submit(query)
        return q.all().result()

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """
        Take GraphDocument as input as uses it to construct a graph.
        """
        node_cache: Dict[Union[str, int], Node] = {}
        for document in graph_documents:
            if include_source:
                # Create document vertex
                doc_props = {
                    "page_content": document.source.page_content,
                    "metadata": document.source.metadata,
                }
                doc_id = hashlib.md5(document.source.page_content.encode()).hexdigest()
                doc_node = self.add_node(
                    Node(id=doc_id, type="Document", properties=doc_props), node_cache
                )

            # Import nodes to vertices
            for n in document.nodes:
                node = self.add_node(n)
                if include_source:
                    # Add Edge to document for each node
                    self.add_edge(
                        Relationship(
                            type="contains information about",
                            source=doc_node,
                            target=node,
                            properties={},
                        )
                    )
                    self.add_edge(
                        Relationship(
                            type="is extracted from",
                            source=node,
                            target=doc_node,
                            properties={},
                        )
                    )

            # Edges
            for el in document.relationships:
                # Find or create the source vertex
                self.add_node(el.source, node_cache)
                # Find or create the target vertex
                self.add_node(el.target, node_cache)
                # Find or create the edge
                self.add_edge(el)

    def build_vertex_query(self, node: Node) -> str:
        base_query = (
            f"g.V().has('id','{node.id}').fold()"
            + f".coalesce(unfold(),addV('{node.type}')"
            + f".property('id','{node.id}')"
            + f".property('type','{node.type}')"
        )
        for key, value in node.properties.items():
            base_query += f".property('{key}', '{value}')"

        return base_query + ")"

    def build_edge_query(self, relationship: Relationship) -> str:
        source_query = f".has('id','{relationship.source.id}')"
        target_query = f".has('id','{relationship.target.id}')"

        base_query = f""""g.V(){source_query}.as('a')  
            .V(){target_query}.as('b') 
            .choose(
                __.inE('{relationship.type}').where(outV().as('a')),
                __.identity(),
                __.addE('{relationship.type}').from('a').to('b')
            )        
            """.replace("\n", "").replace("\t", "")
        for key, value in relationship.properties.items():
            base_query += f".property('{key}', '{value}')"

        return base_query

    def add_node(self, node: Node, node_cache: dict = {}) -> Node:
        # if properties does not have label, add type as label
        if "label" not in node.properties:
            node.properties["label"] = node.type
        if node.id in node_cache:
            return node_cache[node.id]
        else:
            query = self.build_vertex_query(node)
            _ = self.client.submit(query).all().result()[0]
            node_cache[node.id] = node
            return node

    def add_edge(self, relationship: Relationship) -> Any:
        query = self.build_edge_query(relationship)
        return self.client.submit(query).all().result()
