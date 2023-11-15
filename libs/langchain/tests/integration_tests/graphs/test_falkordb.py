import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from langchain.graphs import FalkorDBGraph


class TestFalkorDB(unittest.TestCase):
    def setUp(self) -> None:
        self.host = "localhost"
        self.graph = "test_falkordb"
        self.port = 6379

    @patch("redis.Redis")
    def test_init(self, mock_client: Any) -> None:
        mock_client.return_value = MagicMock()
        FalkorDBGraph(database=self.graph, host=self.host, port=self.port)

    @patch("redis.Redis")
    def test_execute(self, mock_client: Any) -> None:
        mock_client.return_value = MagicMock()
        graph = FalkorDBGraph(database=self.graph, host=self.host, port=self.port)

        query = "RETURN 1"
        result = graph.query(query)
        self.assertIsInstance(result, MagicMock)

    @patch("redis.Redis")
    def test_refresh_schema(self, mock_client: Any) -> None:
        mock_client.return_value = MagicMock()
        graph = FalkorDBGraph(database=self.graph, host=self.host, port=self.port)

        graph.refresh_schema()
        self.assertNotEqual(graph.get_schema, "")
