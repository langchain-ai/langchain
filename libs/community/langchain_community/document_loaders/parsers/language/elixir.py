from typing import TYPE_CHECKING

from langchain_community.document_loaders.parsers.language.tree_sitter_segmenter import (  # noqa: E501
    TreeSitterSegmenter,
)

if TYPE_CHECKING:
    from tree_sitter import Language, Parser

CHUNK_QUERY = """
    [
        (unary_operator
            operator: "@"
            operand: (call
                target: (identifier)
                (arguments
                    [
                        (string)
                        (charlist)
                        (sigil
                            quoted_start: _
                            quoted_end: _
                        )
                        (boolean)
                    ]
                )
            )
        ) @comment
    
        (call
            target: (identifier)
            (arguments (alias))
        ) @module
    
        (call
            target: (identifier)
            (arguments
                [
                    ; zero-arity functions with no parentheses
                    (identifier)
                    ; regular function clause
                    (call target: (identifier))
                    ; function clause with a guard clause
                    (binary_operator
                        left: (call target: (identifier))
                        operator: "when"
                    )
                ]
            )
        ) @function
    ]
""".strip()


class ElixirSegmenter(TreeSitterSegmenter):
    """Code segmenter for Elixir."""

    def get_language(self) -> "Language":
        from tree_sitter_language_pack import get_language

        return get_language("elixir")

    def get_parser(self) -> "Parser":
        from tree_sitter_language_pack import get_parser

        return get_parser("elixir")

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY

    def make_line_comment(self, text: str) -> str:
        return f"# {text}"
