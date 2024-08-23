from typing import TYPE_CHECKING

from langchain_community.document_loaders.parsers.language.tree_sitter_segmenter import (  # noqa: E501
    TreeSitterSegmenter,
)

if TYPE_CHECKING:
    from tree_sitter import Language


CHUNK_QUERY = """
    [
        (call target: ((identifier) @_identifier
            (#any-of? @_identifier "defmodule" "defprotocol" "defimpl"))) @module
        (call target: ((identifier) @_identifier
            (#any-of? @_identifier "def" "defmacro" "defmacrop" "defp"))) @function
        (unary_operator operator: "@" operand: (call target: ((identifier) @_identifier
              (#any-of? @_identifier "moduledoc" "typedoc""doc")))) @comment
    ]
""".strip()


class ElixirSegmenter(TreeSitterSegmenter):
    """Code segmenter for Elixir."""

    def get_language(self) -> "Language":
        from tree_sitter_languages import get_language

        return get_language("elixir")

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY

    def make_line_comment(self, text: str) -> str:
        return f"# {text}"
