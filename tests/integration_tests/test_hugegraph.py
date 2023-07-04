import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from langchain.graphs import HugeGraph


class TestHugeGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.username = "test_user"
        self.password = "test_password"
        self.address = "test_address"
        self.graph = "test_hugegraph"
        self.port = 1234
        self.session_pool_size = 10

    @patch("hugegraph.connection.PyHugeGraph")
    def test_init(self, mock_client: Any) -> None:
        mock_client.return_value = MagicMock()
        huge_graph = HugeGraph(
            self.username, self.password, self.address, self.port, self.graph
        )
        self.assertEqual(huge_graph.username, self.username)
        self.assertEqual(huge_graph.password, self.password)
        self.assertEqual(huge_graph.address, self.address)
        self.assertEqual(huge_graph.port, self.port)
        self.assertEqual(huge_graph.graph, self.graph)

    @patch("hugegraph.connection.PyHugeGraph")
    def test_execute(self, mock_client: Any) -> None:
        mock_client.return_value = MagicMock()
        huge_graph = HugeGraph(
            self.username, self.password, self.address, self.port, self.graph
        )
        query = "g.V().limit(10)"
        result = huge_graph.query(query)
        self.assertIsInstance(result, MagicMock)

    @patch("hugegraph.connection.PyHugeGraph")
    def test_refresh_schema(self, mock_client: Any) -> None:
        mock_client.return_value = MagicMock()
        huge_graph = HugeGraph(
            self.username, self.password, self.address, self.port, self.graph
        )
        huge_graph.refresh_schema()
        self.assertNotEqual(huge_graph.get_schema, "")
