from typing import Any, Dict, List


class HugeGraph:
    """HugeGraph wrapper for graph operations"""

    def __init__(
        self,
        username: str = "default",
        password: str = "default",
        address: str = "127.0.0.1",
        port: int = 8081,
        graph: str = "hugegraph",
    ) -> None:
        """Create a new HugeGraph wrapper instance."""
        try:
            from hugegraph.connection import PyHugeGraph
        except ImportError:
            raise ValueError(
                "Please install HugeGraph Python client first: "
                "`pip3 install hugegraph-python`"
            )

        self.username = username
        self.password = password
        self.address = address
        self.port = port
        self.graph = graph
        self.client = PyHugeGraph(
            address, port, user=username, pwd=password, graph=graph
        )
        self.schema = ""
        # Set schema
        try:
            self.refresh_schema()
        except Exception as e:
            raise ValueError(f"Could not refresh schema. Error: {e}")

    @property
    def get_schema(self) -> str:
        """Returns the schema of the HugeGraph database"""
        return self.schema

    def refresh_schema(self) -> None:
        """
        Refreshes the HugeGraph schema information.
        """
        schema = self.client.schema()
        vertex_schema = schema.getVertexLabels()
        edge_schema = schema.getEdgeLabels()
        relationships = schema.getRelations()

        self.schema = (
            f"Node properties: {vertex_schema}\n"
            f"Edge properties: {edge_schema}\n"
            f"Relationships: {relationships}\n"
        )

    def query(self, query: str) -> List[Dict[str, Any]]:
        g = self.client.gremlin()
        res = g.exec(query)
        return res["data"]
