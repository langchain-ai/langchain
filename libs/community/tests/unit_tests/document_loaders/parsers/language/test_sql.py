import unittest

import pytest

from langchain_community.document_loaders.parsers.language.sql import SQLSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestSQLSegmenter(unittest.TestCase):
    """Unit tests for the SQLSegmenter class."""

    def setUp(self) -> None:
        """Set up example code and expected results for testing."""
        self.example_code = """
        CREATE TABLE users (id INT, name TEXT);

        -- A select query
        SELECT id, name FROM users WHERE id = 1;

        INSERT INTO users (id, name) VALUES (2, 'Alice');

        UPDATE users SET name = 'Bob' WHERE id = 2;

        DELETE FROM users WHERE id = 2;
        """

        self.expected_simplified_code = (
            "-- Code for: CREATE TABLE users (id INT, name TEXT);\n"
            "-- Code for: SELECT id, name FROM users WHERE id = 1;\n"
            "-- Code for: INSERT INTO users (id, name) VALUES (2, 'Alice');\n"
            "-- Code for: UPDATE users SET name = 'Bob' WHERE id = 2;\n"
            "-- Code for: DELETE FROM users WHERE id = 2;"
        )

        self.expected_extracted_code = [
            "CREATE TABLE users (id INT, name TEXT);",
            "SELECT id, name FROM users WHERE id = 1;",
            "INSERT INTO users (id, name) VALUES (2, 'Alice');",
            "UPDATE users SET name = 'Bob' WHERE id = 2;",
            "DELETE FROM users WHERE id = 2;",
        ]

    def test_is_valid(self) -> None:
        """Test the validity of SQL code."""
        # Valid SQL code should return True
        self.assertTrue(SQLSegmenter("SELECT * FROM test").is_valid())
        # Invalid code (non-SQL text) should return False
        self.assertFalse(SQLSegmenter("random text").is_valid())

    def test_extract_functions_classes(self) -> None:
        """Test extracting SQL statements from code."""
        segmenter = SQLSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        # Verify the extracted code matches expected SQL statements
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        """Test simplifying SQL code into commented descriptions."""
        segmenter = SQLSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
