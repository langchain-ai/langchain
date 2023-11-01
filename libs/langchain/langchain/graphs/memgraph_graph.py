from langchain.graphs.neo4j_graph import Neo4jGraph

SCHEMA_QUERY = """
CALL llm_util.schema("prompt_ready")
YIELD *
RETURN *
"""

RAW_SCHEMA_QUERY = """
CALL llm_util.schema("raw")
YIELD *
RETURN *
"""


class MemgraphGraph(Neo4jGraph):
    """Memgraph wrapper for graph operations.

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
        self, url: str, username: str, password: str, *, database: str = "memgraph"
    ) -> None:
        """Create a new Memgraph graph wrapper instance."""
        super().__init__(url, username, password, database=database)

    def refresh_schema(self) -> None:
        """
        Refreshes the Memgraph graph schema information.
        """

        db_schema = self.query(SCHEMA_QUERY)[0].get("schema")
        assert db_schema is not None
        self.schema = db_schema

        db_structured_schema = self.query(RAW_SCHEMA_QUERY)[0].get("schema")
        assert db_structured_schema is not None
        self.structured_schema = db_structured_schema
