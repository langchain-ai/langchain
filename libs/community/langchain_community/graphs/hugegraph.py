from typing import Any, Dict, List


class HugeGraph:
    """HugeGraph wrapper for graph operations.

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
            raise ImportError(
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
