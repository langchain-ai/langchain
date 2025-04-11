import unittest
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

from sqlalchemy import Column, Integer, MetaData, String, Table

from langchain_community.utilities.sql_database import SQLDatabase


class TestSQLDatabaseComments(unittest.TestCase):
    """Test class for column comment functionality in SQLDatabase"""

    def setUp(self) -> None:
        """Setup before each test"""
        # Mock Engine and actual connection
        self.mock_engine = MagicMock()
        self.mock_engine.dialect.name = "postgresql"  # Default to PostgreSQL

        # Mock inspector
        self.mock_inspector = MagicMock()
        self.patch_inspector = patch(
            "sqlalchemy.inspect", return_value=self.mock_inspector
        )
        self.mock_inspect = self.patch_inspector.start()

        # Mock table name list
        self.mock_inspector.get_table_names.return_value = ["test_table"]
        self.mock_inspector.get_view_names.return_value = []
        self.mock_inspector.get_indexes.return_value = []

        # Mock metadata
        self.metadata = MetaData()

        # Create test database object
        self.db = SQLDatabase(engine=self.mock_engine, metadata=self.metadata)

    def tearDown(self) -> None:
        """Cleanup after each test"""
        self.patch_inspector.stop()

    def setup_mock_table_with_comments(
        self, dialect: str, comments: Optional[Dict[str, str]] = None
    ) -> Table:
        """Setup a mock table with comments

        Args:
            dialect (str): Database dialect to test (postgresql, mysql, oracle)
            comments (dict, optional): Column comments. Uses default comments if None

        Returns:
            Table: The created mock table
        """
        # Default comments
        if comments is None:
            comments = {
                "id": "Primary key",
                "name": "Name of the person",
                "age": "Age of the person",
            }

        # Set engine dialect
        self.mock_engine.dialect.name = dialect

        # Create test table
        test_table = Table(
            "test_table",
            self.metadata,
            Column("id", Integer, primary_key=True, comment=comments.get("id")),
            Column("name", String(100), comment=comments.get("name")),
            Column("age", Integer, comment=comments.get("age")),
        )

        # Mock table compilation function.
        mock_create_table = MagicMock(
            return_value=(
                "CREATE TABLE test_table (\n\tid INTEGER PRIMARY KEY,"
                "\n\tname VARCHAR(100),\n\tage INTEGER\n)"
            )
        )
        with patch("sqlalchemy.schema.CreateTable.compile", mock_create_table):
            pass

        # Insert table into metadata (mocking internal SQLAlchemy behavior)
        self.metadata._add_table("test_table", None, test_table)

        return test_table

    def test_postgres_get_col_comments(self) -> None:
        """Test retrieving column comments from PostgreSQL"""
        # Setup PostgreSQL table with comments
        self.setup_mock_table_with_comments("postgresql")

        # Call get_table_info with get_col_comments=True
        table_info = self.db.get_table_info(get_col_comments=True)

        # Verify comments are included in table info
        self.assertIn("Column 'id': Primary key", table_info)
        self.assertIn("Column 'name': Name of the person", table_info)
        self.assertIn("Column 'age': Age of the person", table_info)

    def test_mysql_get_col_comments(self) -> None:
        """Test retrieving column comments from MySQL"""
        # Setup MySQL table with comments
        self.setup_mock_table_with_comments("mysql")

        # Call get_table_info with get_col_comments=True
        table_info = self.db.get_table_info(get_col_comments=True)

        # Verify comments are included in table info
        self.assertIn("Column 'id': Primary key", table_info)
        self.assertIn("Column 'name': Name of the person", table_info)
        self.assertIn("Column 'age': Age of the person", table_info)

    def test_oracle_get_col_comments(self) -> None:
        """Test retrieving column comments from Oracle"""
        # Setup Oracle table with comments
        self.setup_mock_table_with_comments("oracle")

        # Call get_table_info with get_col_comments=True
        table_info = self.db.get_table_info(get_col_comments=True)

        # Verify comments are included in table info
        self.assertIn("Column 'id': Primary key", table_info)
        self.assertIn("Column 'name': Name of the person", table_info)
        self.assertIn("Column 'age': Age of the person", table_info)

    def test_sqlite_no_comments(self) -> None:
        """Test that SQLite does not support column comments"""
        # Setup SQLite table (without comments)
        self.setup_mock_table_with_comments("sqlite", comments={})

        # Check that an exception is raised when calling get_table_info
        with self.assertRaises(ValueError) as context:
            self.db.get_table_info(get_col_comments=True)

        # Verify exception message
        self.assertIn(
            "Column comments are available on PostgreSQL, MySQL, Oracle",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
