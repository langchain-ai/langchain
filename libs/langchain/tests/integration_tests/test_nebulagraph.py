import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from langchain.graphs import NebulaGraph


class TestNebulaGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.space = "test_space"
        self.username = "test_user"
        self.password = "test_password"
        self.address = "test_address"
        self.port = 1234
        self.session_pool_size = 10

    @patch("nebula3.gclient.net.SessionPool.SessionPool")
    def test_init(self, mock_session_pool: Any) -> None:
        mock_session_pool.return_value = MagicMock()
        nebula_graph = NebulaGraph(
            self.space,
            self.username,
            self.password,
            self.address,
            self.port,
            self.session_pool_size,
        )
        self.assertEqual(nebula_graph.space, self.space)
        self.assertEqual(nebula_graph.username, self.username)
        self.assertEqual(nebula_graph.password, self.password)
        self.assertEqual(nebula_graph.address, self.address)
        self.assertEqual(nebula_graph.port, self.port)
        self.assertEqual(nebula_graph.session_pool_size, self.session_pool_size)

    @patch("nebula3.gclient.net.SessionPool.SessionPool")
    def test_get_session_pool(self, mock_session_pool: Any) -> None:
        mock_session_pool.return_value = MagicMock()
        nebula_graph = NebulaGraph(
            self.space,
            self.username,
            self.password,
            self.address,
            self.port,
            self.session_pool_size,
        )
        session_pool = nebula_graph._get_session_pool()
        self.assertIsInstance(session_pool, MagicMock)

    @patch("nebula3.gclient.net.SessionPool.SessionPool")
    def test_del(self, mock_session_pool: Any) -> None:
        mock_session_pool.return_value = MagicMock()
        nebula_graph = NebulaGraph(
            self.space,
            self.username,
            self.password,
            self.address,
            self.port,
            self.session_pool_size,
        )
        nebula_graph.__del__()
        mock_session_pool.return_value.close.assert_called_once()

    @patch("nebula3.gclient.net.SessionPool.SessionPool")
    def test_execute(self, mock_session_pool: Any) -> None:
        mock_session_pool.return_value = MagicMock()
        nebula_graph = NebulaGraph(
            self.space,
            self.username,
            self.password,
            self.address,
            self.port,
            self.session_pool_size,
        )
        query = "SELECT * FROM test_table"
        result = nebula_graph.execute(query)
        self.assertIsInstance(result, MagicMock)

    @patch("nebula3.gclient.net.SessionPool.SessionPool")
    def test_refresh_schema(self, mock_session_pool: Any) -> None:
        mock_session_pool.return_value = MagicMock()
        nebula_graph = NebulaGraph(
            self.space,
            self.username,
            self.password,
            self.address,
            self.port,
            self.session_pool_size,
        )
        nebula_graph.refresh_schema()
        self.assertNotEqual(nebula_graph.get_schema, "")
