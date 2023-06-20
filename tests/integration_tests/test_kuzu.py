import unittest
from typing import Any
from langchain.graphs import KuzuGraph
import tempfile
import shutil

EXPECTED_SCHEMA = """
Node properties: [{'properties': [('name', 'STRING')], 'label': 'Movie'}, {'properties': [('name', 'STRING'), ('birthDate', 'STRING')], 'label': 'Person'}]
Relationships properties: [{'properties': [], 'label': 'ActedIn'}]
Relationships: ['(:Person)-[:ActedIn]->(:Movie)']
"""


class TestKuzu(unittest.TestCase):
    def setUp(self) -> None:
        import kuzu
        self.tmpdir = tempfile.mkdtemp()
        self.kuzu_database = kuzu.Database(self.tmpdir)
        self.conn = kuzu.Connection(self.kuzu_database)
        self.conn.execute("CREATE NODE TABLE Movie (name STRING, PRIMARY KEY(name))")
        self.conn.execute("CREATE (:Movie {name: 'The Godfather'})")
        self.conn.execute("CREATE (:Movie {name: 'The Godfather: Part II'})")
        self.conn.execute(
            "CREATE (:Movie {name: 'The Godfather Coda: The Death of Michael Corleone'})")
        self.kuzu_graph = KuzuGraph(self.kuzu_database)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_query(self) -> None:
        result = self.kuzu_graph.query("MATCH (n:Movie) RETURN n.name ORDER BY n.name")
        excepted_result = [{'n.name': 'The Godfather'},
                           {'n.name': 'The Godfather Coda: The Death of Michael Corleone'},
                           {'n.name': 'The Godfather: Part II'}]
        self.assertEqual(result, excepted_result)

    def test_refresh_schema(self) -> None:
        self.conn.execute(
            "CREATE NODE TABLE Person (name STRING, birthDate STRING, PRIMARY KEY(name))")
        self.conn.execute("CREATE REL TABLE ActedIn (FROM Person TO Movie)")
        self.kuzu_graph.refresh_schema()
        schema = self.kuzu_graph.get_schema
        self.assertEqual(schema, EXPECTED_SCHEMA)
