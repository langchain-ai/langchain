import unittest

import pytest

from langchain_community.document_loaders.parsers.language.sql import SQLSegmenter


@pytest.mark.requires("tree_sitter", "tree_sitter_languages")
class TestSQLSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.example_code = """
        CREATE TABLE users (id INT, name TEXT);

        -- A select query
        SELECT id, name FROM users WHERE id = 1;

        INSERT INTO users (id, name) VALUES (2, 'Alice');

        UPDATE users SET name = 'Bob' WHERE id = 2;

        DELETE FROM users WHERE id = 2;
        """

        self.expected_simplified_code = """

         -- Code for: CREATE TABLE users (id INT, name TEXT);
-- Code for: SELECT id, name FROM users WHERE id = 1;
-- Code for: INSERT INTO users (id, name) VALUES (2, 'Alice');
-- Code for: UPDATE users SET name = 'Bob' WHERE id = 2;
-- Code for: DELETE FROM users WHERE id = 2;"""

        self.expected_extracted_code = [
            "CREATE TABLE users (id INT, name TEXT);",
            "SELECT id, name FROM users WHERE id = 1;",
            "INSERT INTO users (id, name) VALUES (2, 'Alice');",
            "UPDATE users SET name = 'Bob' WHERE id = 2;",
            "DELETE FROM users WHERE id = 2;",
        ]

    def test_is_valid(self) -> None:
        self.assertTrue(SQLSegmenter("SELECT * FROM test").is_valid())
        self.assertFalse(SQLSegmenter("random text").is_valid())

    def test_extract_functions_classes(self) -> None:
        segmenter = SQLSegmenter(self.example_code)
        extracted_code = segmenter.extract_functions_classes()
        self.assertEqual(extracted_code, self.expected_extracted_code)

    def test_simplify_code(self) -> None:
        segmenter = SQLSegmenter(self.example_code)
        simplified_code = segmenter.simplify_code()
        self.assertEqual(simplified_code, self.expected_simplified_code)
