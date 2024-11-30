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
    """Code segmenter for SQL."""

    def get_language(self) -> "Language":
        from tree_sitter_languages import get_language

        return get_language("sql")

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY

    def extract_functions_classes(self) -> list[str]:
        """Extract SQL statements from the code."""
        extracted = super().extract_functions_classes()
        # Ensure all statements end with a semicolon
        return [
            stmt.strip() + ";" if not stmt.strip().endswith(";") else stmt.strip()
            for stmt in extracted
        ]

    def simplify_code(self) -> str:
        """Simplify the extracted SQL code into comments."""
        return "\n".join(
            [
                f"-- Code for: {stmt.strip()}"
                for stmt in self.extract_functions_classes()
            ]
        )

    def make_line_comment(self, text: str) -> str:
        return f"-- {text}"
