import shutil
import tempfile
import unittest

from langchain_community.graphs import KuzuGraph

EXPECTED_SCHEMA = """Node properties: [{'properties': [('name', 'STRING')], 'label': 'Movie'}, {'properties': [('name', 'STRING'), ('birthDate', 'STRING')], 'label': 'Person'}]
Relationships properties: [{'properties': [], 'label': 'ActedIn'}]
Relationships: ['(:Person)-[:ActedIn]->(:Movie)']
"""  # noqa: E501


class TestKuzu(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import kuzu
        except ImportError as e:
            raise ImportError(
                "Cannot import Python package kuzu. Please install it by running "
                "`pip install kuzu`."
            ) from e

        self.tmpdir = tempfile.mkdtemp()
        self.kuzu_database = kuzu.Database(self.tmpdir)
        self.conn = kuzu.Connection(self.kuzu_database)
        self.conn.execute("CREATE NODE TABLE Movie (name STRING, PRIMARY KEY(name))")
        self.conn.execute("CREATE (:Movie {name: 'The Godfather'})")
        self.conn.execute("CREATE (:Movie {name: 'The Godfather: Part II'})")
        self.conn.execute(
            "CREATE (:Movie {name: 'The Godfather Coda: The Death of Michael "
            "Corleone'})"
        )
        self.kuzu_graph = KuzuGraph(self.kuzu_database)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_query_no_params(self) -> None:
        result = self.kuzu_graph.query("MATCH (n:Movie) RETURN n.name ORDER BY n.name")
        excepted_result = [
            {"n.name": "The Godfather"},
            {"n.name": "The Godfather Coda: The Death of Michael Corleone"},
            {"n.name": "The Godfather: Part II"},
        ]
        self.assertEqual(result, excepted_result)

    def test_query_params(self) -> None:
        result = self.kuzu_graph.query(
            query="MATCH (n:Movie) WHERE n.name = $name RETURN n.name",
            params={"name": "The Godfather"},
        )
        excepted_result = [
            {"n.name": "The Godfather"},
        ]
        self.assertEqual(result, excepted_result)

    def test_refresh_schema(self) -> None:
        self.conn.execute(
            "CREATE NODE TABLE Person (name STRING, birthDate STRING, PRIMARY "
            "KEY(name))"
        )
        self.conn.execute("CREATE REL TABLE ActedIn (FROM Person TO Movie)")
        self.kuzu_graph.refresh_schema()
        schema = self.kuzu_graph.get_schema
        self.assertEqual(schema, EXPECTED_SCHEMA)
