from typing import Any, Dict, List

from langchain.graphs import Neo4jGraph

schema_query = """
CALL llm_util.schema("prompt_ready")
YIELD *
RETURN *
"""


class MemgraphGraph(Neo4jGraph):
    """Memgraph wrapper for graph operations."""

    def __init__(
        self, url: str, username: str, password: str, database: str = "memgraph"
    ) -> None:
        """Create a new Memgraph graph wrapper instance."""
        try:
            import neo4j
        except ImportError:
            raise ValueError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )

        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        self._database = database
        self.schema = ""
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
        self.refresh_schema()

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Memgraph database"""
        return self.schema

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Memgraph database."""
        from neo4j.exceptions import CypherSyntaxError

        with self._driver.session(database=self._database) as session:
            try:
                data = session.run(query, params)
                # Hard limit of 50 results
                return [r.data() for r in data][:50]
            except CypherSyntaxError as e:
                raise ValueError(f"Generated Cypher Statement is not valid\n{e}")

    def refresh_schema(self) -> None:
        """
        Refreshes the Memgraph graph schema information.
        """

        db_schema = self.query(schema_query)[0].get("schema")
        assert db_schema is not None
        self.schema = db_schema
