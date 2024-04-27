from collections import namedtuple
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_community.utilities.cassandra_database import (
    CassandraDatabase,
    DatabaseError,
    Table,
)

# Define a namedtuple type
MockRow = namedtuple("MockRow", ["col1", "col2"])


class TestCassandraDatabase(object):
    def setup_method(self) -> None:
        self.mock_session = MagicMock()
        self.cassandra_db = CassandraDatabase(session=self.mock_session)

    def test_init_without_session(self) -> None:
        with pytest.raises(ValueError):
            CassandraDatabase()

    def test_run_query(self) -> None:
        # Mock the execute method to return an iterable of dictionaries directly
        self.mock_session.execute.return_value = iter(
            [{"col1": "val1", "col2": "val2"}]
        )

        # Execute the query
        result = self.cassandra_db.run("SELECT * FROM table")

        # Assert that the result is as expected
        assert result == [{"col1": "val1", "col2": "val2"}]

        # Verify that execute was called with the expected CQL query
        self.mock_session.execute.assert_called_with("SELECT * FROM table")

    def test_run_query_cursor(self) -> None:
        mock_result_set = MagicMock()
        self.mock_session.execute.return_value = mock_result_set
        result = self.cassandra_db.run("SELECT * FROM table;", fetch="cursor")
        assert result == mock_result_set

    def test_run_query_invalid_fetch(self) -> None:
        with pytest.raises(ValueError):
            self.cassandra_db.run("SELECT * FROM table;", fetch="invalid")

    def test_validate_cql_select(self) -> None:
        query = "SELECT * FROM table;"
        result = self.cassandra_db._validate_cql(query, "SELECT")
        assert result == "SELECT * FROM table"

    def test_validate_cql_unsupported_type(self) -> None:
        query = "UPDATE table SET col=val;"
        with pytest.raises(ValueError):
            self.cassandra_db._validate_cql(query, "UPDATE")

    def test_validate_cql_unsafe(self) -> None:
        query = "SELECT * FROM table; DROP TABLE table;"
        with pytest.raises(DatabaseError):
            self.cassandra_db._validate_cql(query, "SELECT")

    @patch(
        "langchain_community.utilities.cassandra_database.CassandraDatabase._resolve_schema"
    )
    def test_format_schema_to_markdown(self, mock_resolve_schema: Any) -> None:
        mock_table1 = MagicMock(spec=Table)
        mock_table1.as_markdown.return_value = "## Keyspace: keyspace1"
        mock_table2 = MagicMock(spec=Table)
        mock_table2.as_markdown.return_value = "## Keyspace: keyspace2"
        mock_resolve_schema.return_value = {
            "keyspace1": [mock_table1],
            "keyspace2": [mock_table2],
        }
        markdown = self.cassandra_db.format_schema_to_markdown()
        assert markdown.startswith("# Cassandra Database Schema")
        assert "## Keyspace: keyspace1" in markdown
        assert "## Keyspace: keyspace2" in markdown


if __name__ == "__main__":
    pytest.main()
