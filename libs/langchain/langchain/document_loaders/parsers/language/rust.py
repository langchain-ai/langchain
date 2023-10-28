from langchain.document_loaders.parsers.language.tree_sitter_segmenter import (
    TreeSitterSegmenter,
)

CHUNK_QUERY = """
    [
        (function_item
            name: (identifier)
            body: (block)) @function
        (struct_item) @struct
        (trait_item) @trait
    ]
""".strip()


class RustSegmenter(TreeSitterSegmenter):
    """Code segmenter for Rust."""

    def get_language(self):
        from tree_sitter_languages import get_language

        return get_language("rust")

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY

    def make_line_comment(self, text: str) -> str:
        return f"// {text}"
