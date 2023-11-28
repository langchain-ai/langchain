import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from langchain.graphs import DGraph


class TestDGraphDB(unittest.TestCase):
    def setUp(self) -> None:
        self.host = "localhost:9080"

    @patch("pydgraph.DgraphClientStub")
    def test_init(self, mock_client: Any) -> None:
        mock_client.return_value = MagicMock()
        dgraph = DGraph(clientUrl=self.host)
        self.assertEqual(dgraph.clientUrl, self.host)

    @patch("pydgraph.DgraphClientStub")
    def test_execute(self, mock_client: Any) -> None:
        mock_client.return_value = MagicMock()
        dgraph = DGraph(clientUrl=self.host)
        query = "{ me(func: has(name)) { name }}}"
        result = dgraph.query(query)
        self.assertIsInstance(result, MagicMock)

    @patch("pydgraph.DgraphClientStub")
    def test_get_schema(self, mock_client: Any) -> None:
        mock_client.return_value = MagicMock()
        dgraph = DGraph(clientUrl=self.host)
        schema = dgraph.get_schema()
        self.assertIsInstance(schema, dict)

    @patch("pydgraph.DgraphClientStub")
    def test_add_schema(self, mock_client: Any) -> None:
        mock_client.return_value = MagicMock()
        dgraph = DGraph(clientUrl=self.host)
        schema = "name: string @index(exact) ."
        dgraph.add_schema(schema)
        self.assertEqual(
            dgraph.get_schema(),
            {"name": [{"predicate": "name", "type": "string", "index": "exact"}]},
        )

    @patch("pydgraph.DgraphClientStub")
    def test_add_node(self, mock_client: Any) -> None:
        mock_client.return_value = MagicMock()
        dgraph = DGraph(clientUrl=self.host)
        data = {"name": "Alice"}
        dgraph.add_node(data)
        query = "{ me(func: has(name)) { name }}}"
        result = dgraph.query(query)
        self.assertEqual(result[0], {"me": [{"name": "Alice"}]})
