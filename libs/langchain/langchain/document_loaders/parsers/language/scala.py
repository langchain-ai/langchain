from langchain.document_loaders.parsers.language.tree_sitter_segmenter import (
    TreeSitterSegmenter,
)

CHUNK_QUERY = """
    [
        (class_definition 
            name: (identifier)) @class
        (function_definition
            name: (identifier)) @function
        (object_definition
            name: (identifier)) @object
        (trait_definition
            name: (identifier)) @trait
    ]
""".strip()


class ScalaSegmenter(TreeSitterSegmenter):
    """Code segmenter for Scala."""

    def get_language(self):
        from tree_sitter_languages import get_language

        return get_language("scala")

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY

    def make_line_comment(self, text: str) -> str:
        return f"// {text}"
