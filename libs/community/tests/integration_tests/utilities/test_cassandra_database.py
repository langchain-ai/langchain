import unittest
from unittest.mock import MagicMock, patch

from cassandra.cluster import ResultSet

from langchain_community.utilities.cassandra_database import (
    CassandraDatabase,
    DatabaseError,
    Table,
)


class TestCassandraDatabase(unittest.TestCase):

    def setUp(self):
        self.mock_session = MagicMock()
        self.cassandra_db = CassandraDatabase(session=self.mock_session)

    def test_init_without_session(self):
        with self.assertRaises(ValueError):
            CassandraDatabase()

    def test_run_query(self):
        self.mock_session.execute.return_value = [{'col1': 'val1', 'col2': 'val2'}]
        result = self.cassandra_db.run("SELECT * FROM table;")
        self.assertEqual(result, [{'col1': 'val1', 'col2': 'val2'}])
        self.mock_session.execute.assert_called_with("SELECT * FROM table;")

    def test_run_query_one(self):
        self.mock_session.execute.return_value = ResultSet([{'col1': 'val1', 'col2': 'val2'}])
        result = self.cassandra_db.run("SELECT * FROM table;", fetch="one")
        self.assertEqual(result, {'col1': 'val1', 'col2': 'val2'})

    def test_run_query_cursor(self):
        mock_result_set = MagicMock()
        self.mock_session.execute.return_value = mock_result_set
        result = self.cassandra_db.run("SELECT * FROM table;", fetch="cursor")
        self.assertEqual(result, mock_result_set)

    def test_run_query_invalid_fetch(self):
        with self.assertRaises(ValueError):
            self.cassandra_db.run("SELECT * FROM table;", fetch="invalid")

    def test_validate_cql_select(self):
        query = "SELECT * FROM table;"
        result = self.cassandra_db._validate_cql(query, "SELECT")
        self.assertEqual(result, "SELECT * FROM table")

    def test_validate_cql_unsupported_type(self):
        query = "UPDATE table SET col=val;"
        with self.assertRaises(ValueError):
            self.cassandra_db._validate_cql(query, "UPDATE")

    def test_validate_cql_unsafe(self):
        query = "SELECT * FROM table; DROP TABLE table;"
        with self.assertRaises(DatabaseError):
            self.cassandra_db._validate_cql(query, "SELECT")

    @patch('your_module.CassandraDatabase._resolve_schema')
    def test_format_schema_to_markdown(self, mock_resolve_schema):
        mock_resolve_schema.return_value = {
            'keyspace1': [MagicMock(spec=Table)],
            'keyspace2': [MagicMock(spec=Table)]
        }
        markdown = self.cassandra_db.format_schema_to_markdown()
        self.assertTrue(markdown.startswith("# Cassandra Database Schema"))
        self.assertIn("## Keyspace: keyspace1", markdown)
        self.assertIn("## Keyspace: keyspace2", markdown)

if __name__ == '__main__':
    unittest.main()