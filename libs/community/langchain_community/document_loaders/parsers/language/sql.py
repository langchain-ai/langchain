from typing import TYPE_CHECKING

from langchain_community.document_loaders.parsers.language.tree_sitter_segmenter import (  # noqa: E501
    TreeSitterSegmenter,
)

if TYPE_CHECKING:
    from tree_sitter import Language

CHUNK_QUERY = """
    [
        (create_table_statement) @create
        (select_statement) @select
        (insert_statement) @insert
        (update_statement) @update
        (delete_statement) @delete
    ]
"""


class SQLSegmenter(TreeSitterSegmenter):
    """Code segmenter for SQL.
    This class uses Tree-sitter to segment SQL code into its
    constituent statements (e.g., SELECT, CREATE TABLE).
    It also provides functionality to extract these
    statements and simplify the code into commented descriptions.
    """

    def get_language(self) -> "Language":
        """Return the SQL language grammar for Tree-sitter."""
        from tree_sitter_languages import get_language

        return get_language("sql")

    def get_chunk_query(self) -> str:
        """Return the Tree-sitter query for SQL segmentation."""
        return CHUNK_QUERY

    def extract_functions_classes(self) -> list[str]:
        """Extract SQL statements from the code.
        Ensures that all SQL statements end with a semicolon
        for consistency.
        """
        extracted = super().extract_functions_classes()
        # Ensure all statements end with a semicolon
        return [
            stmt.strip() + ";" if not stmt.strip().endswith(";") else stmt.strip()
            for stmt in extracted
        ]

    def simplify_code(self) -> str:
        """Simplify the extracted SQL code into comments.
        Converts SQL statements into commented descriptions
        for easy readability.
        """
        return "\n".join(
            [
                f"-- Code for: {stmt.strip()}"
                for stmt in self.extract_functions_classes()
            ]
        )

    def make_line_comment(self, text: str) -> str:
        """Create a line comment in SQL style."""
        return f"-- {text}"
