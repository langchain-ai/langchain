from typing import Any, Dict, List, Optional

from langchain_community.graphs.graph_store import GraphStore


class TigerGraph(GraphStore):
    """TigerGraph wrapper for graph operations.

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

    def __init__(self, conn: Any) -> None:
        """Create a new TigerGraph graph wrapper instance."""
        self.set_connection(conn)
        self.set_schema()

    @property
    def conn(self) -> Any:
        return self._conn

    @property
    def schema(self) -> Dict[str, Any]:
        return self._schema

    def get_schema(self) -> str:  # type: ignore[override]
        if self._schema:
            return str(self._schema)
        else:
            self.set_schema()
            return str(self._schema)

    def set_connection(self, conn: Any) -> None:
        try:
            from pyTigerGraph import TigerGraphConnection
        except ImportError:
            raise ImportError(
                "Could not import pyTigerGraph python package. "
                "Please install it with `pip install pyTigerGraph`."
            )

        if not isinstance(conn, TigerGraphConnection):
            msg = "**conn** parameter must inherit from TigerGraphConnection"
            raise TypeError(msg)

        if conn.ai.nlqs_host is None:
            msg = """**conn** parameter does not have nlqs_host parameter defined.
                     Define hostname of NLQS service."""
            raise ConnectionError(msg)

        self._conn: TigerGraphConnection = conn
        self.set_schema()

    def set_schema(self, schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Set the schema of the TigerGraph Database.
        Auto-generates Schema if **schema** is None.
        """
        self._schema = self.generate_schema() if schema is None else schema

    def generate_schema(
        self,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates the schema of the TigerGraph Database and returns it
        User can specify a **sample_ratio** (0 to 1) to determine the
        ratio of documents/edges used (in relation to the Collection size)
        to render each Collection Schema.
        """
        return self._conn.getSchema(force=True)

    def refresh_schema(self) -> None:
        self.generate_schema()

    def query(self, query: str) -> Dict[str, Any]:  # type: ignore[override]
        """Query the TigerGraph database."""
        answer = self._conn.ai.query(query)
        return answer

    def register_query(
        self,
        function_header: str,
        description: str,
        docstring: str,
        param_types: dict = {},
    ) -> List[str]:
        """
        Wrapper function to register a custom GSQL query to the TigerGraph NLQS.
        """
        return self._conn.ai.registerCustomQuery(
            function_header, description, docstring, param_types
        )
