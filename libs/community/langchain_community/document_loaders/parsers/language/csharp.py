from typing import TYPE_CHECKING

from langchain_community.document_loaders.parsers.language.tree_sitter_segmenter import (  # noqa: E501
    TreeSitterSegmenter,
)

if TYPE_CHECKING:
    from tree_sitter import Language


CHUNK_QUERY = """
    [
        (namespace_declaration) @namespace
        (class_declaration) @class
        (method_declaration) @method
        (interface_declaration) @interface
        (enum_declaration) @enum
        (struct_declaration) @struct
        (record_declaration) @record
    ]
""".strip()


class CSharpSegmenter(TreeSitterSegmenter):
    """Code segmenter for C#."""

    def get_language(self) -> "Language":
        from tree_sitter_languages import get_language

        return get_language("c_sharp")

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY

    def make_line_comment(self, text: str) -> str:
        return f"// {text}"
