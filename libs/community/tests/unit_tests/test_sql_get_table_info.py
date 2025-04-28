import unittest
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

from sqlalchemy import Column, Integer, MetaData, String, Table

from langchain_community.utilities.sql_database import SQLDatabase


class TestSQLDatabaseComments(unittest.TestCase):
    """Test class for column comment functionality in SQLDatabase"""

    def setUp(self) -> None:
        """Setup before each test"""
        # Mock Engine
        self.mock_engine = MagicMock()
        self.mock_engine.dialect.name = "postgresql"  # Default to PostgreSQL

        # Mock inspector and start patch *before* SQLDatabase initialization
        self.mock_inspector = MagicMock()
        # Mock table name list and other inspector methods called during init
        self.mock_inspector.get_table_names.return_value = ["test_table"]
        self.mock_inspector.get_view_names.return_value = []
        self.mock_inspector.get_indexes.return_value = []
        # Mock get_columns to return something reasonable for reflection
        self.mock_inspector.get_columns.return_value = [
            {
                "name": "id",
                "type": Integer(),
                "nullable": False,
                "default": None,
                "autoincrement": "auto",
                "comment": None,
            },
            {
                "name": "name",
                "type": String(100),
                "nullable": True,
                "default": None,
                "autoincrement": "auto",
                "comment": None,
            },
            {
                "name": "age",
                "type": Integer(),
                "nullable": True,
                "default": None,
                "autoincrement": "auto",
                "comment": None,
            },
        ]
        # Mock get_pk_constraint for reflection
        self.mock_inspector.get_pk_constraint.return_value = {
            "constrained_columns": ["id"],
            "name": None,
        }
        # Mock get_foreign_keys for reflection
        self.mock_inspector.get_foreign_keys.return_value = []

        # Patch sqlalchemy.inspect to return our mock inspector
        self.patch_inspector = patch(
            "langchain_community.utilities.sql_database.inspect",
            return_value=self.mock_inspector,
        )
        # Start the patch *before* creating the SQLDatabase instance
        self.mock_inspect = self.patch_inspector.start()

        # Mock metadata
        self.metadata = MetaData()

        # Create test database object *after* patching inspect
        try:
            self.db = SQLDatabase(
                engine=self.mock_engine,
                metadata=self.metadata,
                lazy_table_reflection=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception during SQLDatabase init: {e}")

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

        # Clear existing metadata if necessary, or use a fresh MetaData object
        self.metadata.clear()

        # Create test table
        test_table = Table(
            "test_table",
            self.metadata,
            Column("id", Integer, primary_key=True, comment=comments.get("id")),
            Column("name", String(100), comment=comments.get("name")),
            Column("age", Integer, comment=comments.get("age")),
        )

        # Mock reflection to return the columns with comments
        # This is crucial because lazy reflection will call inspect later
        self.mock_inspector.get_columns.return_value = [
            {
                "name": "id",
                "type": Integer(),
                "nullable": False,
                "default": None,
                "autoincrement": "auto",
                "comment": comments.get("id"),
            },
            {
                "name": "name",
                "type": String(100),
                "nullable": True,
                "default": None,
                "autoincrement": "auto",
                "comment": comments.get("name"),
            },
            {
                "name": "age",
                "type": Integer(),
                "nullable": True,
                "default": None,
                "autoincrement": "auto",
                "comment": comments.get("age"),
            },
        ]
        self.mock_inspector.get_table_names.return_value = [
            "test_table"
        ]  # Ensure table is discoverable

        # No need to mock CreateTable here, let the actual code call it.
        # We will patch it during the get_table_info call in the tests.

        # No need to manually add table to metadata, reflection handles it
        # self.metadata._add_table("test_table", None, test_table)

        return test_table

    def _run_test_with_mocked_createtable(self, dialect: str) -> None:
        """Helper function to run comment tests with CreateTable mocked."""
        self.setup_mock_table_with_comments(dialect)

        # Define the expected CREATE TABLE string
        expected_create_table_sql = (
            "CREATE TABLE test_table (\n\tid INTEGER NOT NULL, "
            "\n\tname VARCHAR(100), \n\tage INTEGER, \n\tPRIMARY KEY (id)\n)"
        )

        # Patch CreateTable specifically for the get_table_info call
        with patch(
            "langchain_community.utilities.sql_database.CreateTable"
        ) as MockCreateTable:
            # Mock the compile method to return a specific string
            mock_compiler = MockCreateTable.return_value.compile
            mock_compiler.return_value = expected_create_table_sql

            # Call get_table_info with get_col_comments=True
            table_info = self.db.get_table_info(get_col_comments=True)

        # Verify CREATE TABLE statement (using the mocked value)
        self.assertIn(expected_create_table_sql.strip(), table_info)

        # Verify comments are included in table info in the correct format
        self.assertIn("/*\nColumn Comments:", table_info)
        self.assertIn("'id': 'Primary key'", table_info)
        self.assertIn("'name': 'Name of the person'", table_info)
        self.assertIn("'age': 'Age of the person'", table_info)
        self.assertIn("*/", table_info)

    def test_postgres_get_col_comments(self) -> None:
        """Test retrieving column comments from PostgreSQL"""
        self._run_test_with_mocked_createtable("postgresql")

    def test_mysql_get_col_comments(self) -> None:
        """Test retrieving column comments from MySQL"""
        self._run_test_with_mocked_createtable("mysql")

    def test_oracle_get_col_comments(self) -> None:
        """Test retrieving column comments from Oracle"""
        self._run_test_with_mocked_createtable("oracle")

    def test_sqlite_no_comments(self) -> None:
        """Test that SQLite does not add a comment block when comments are missing."""
        # Setup SQLite table (comments will be ignored by SQLAlchemy for SQLite)
        self.setup_mock_table_with_comments("sqlite", comments={})
        # Mock reflection to return columns *without* comments
        self.mock_inspector.get_columns.return_value = [
            {
                "name": "id",
                "type": Integer(),
                "nullable": False,
                "default": None,
                "autoincrement": "auto",
                "comment": None,
            },
            {
                "name": "name",
                "type": String(100),
                "nullable": True,
                "default": None,
                "autoincrement": "auto",
                "comment": None,
            },
            {
                "name": "age",
                "type": Integer(),
                "nullable": True,
                "default": None,
                "autoincrement": "auto",
                "comment": None,
            },
        ]

        # Define the expected CREATE TABLE string
        expected_create_table_sql = (
            "CREATE TABLE test_table (\n\tid INTEGER NOT NULL, "
            "\n\tname VARCHAR(100), \n\tage INTEGER, \n\tPRIMARY KEY (id)\n)"
        )

        # Patch CreateTable specifically for the get_table_info call
        with patch(
            "langchain_community.utilities.sql_database.CreateTable"
        ) as MockCreateTable:
            mock_compiler = MockCreateTable.return_value.compile
            mock_compiler.return_value = expected_create_table_sql

            # Call get_table_info with get_col_comments=True
            # Even if True, SQLite won't have comments to add.
            table_info = self.db.get_table_info(get_col_comments=True)

        # Verify CREATE TABLE statement
        self.assertIn(expected_create_table_sql.strip(), table_info)
        # Verify comments block is NOT included
        self.assertNotIn("Column Comments:", table_info)


if __name__ == "__main__":
    unittest.main()
