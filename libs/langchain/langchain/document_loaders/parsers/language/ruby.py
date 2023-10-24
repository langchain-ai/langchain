from langchain.document_loaders.parsers.language.tree_sitter_segmenter import (
    TreeSitterSegmenter,
)

CHUNK_QUERY = """
    [
        (method 
            name: (identifier)) @method
        (module 
            name: (constant)) @module
        (class 
            name: (constant)) @class
    ]
""".strip()


class RubySegmenter(TreeSitterSegmenter):
    """Code segmenter for Ruby."""

    def get_language(self):
        from tree_sitter_languages import get_language

        return get_language("ruby")

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY

    def make_line_comment(self, text: str) -> str:
        return f"# {text}"
