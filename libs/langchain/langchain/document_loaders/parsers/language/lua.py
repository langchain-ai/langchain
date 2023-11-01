from langchain.document_loaders.parsers.language.tree_sitter_segmenter import (
    TreeSitterSegmenter,
)

CHUNK_QUERY = """
    [
        (function_definition_statement
            name: (identifier)) @function
        (local_function_definition_statement
            name: (identifier)) @function
    ]
""".strip()

class LuaSegmenter(TreeSitterSegmenter):
    """Code segmenter for Lua."""

    def get_language(self):
        from tree_sitter_languages import get_language

        return get_language("lua")

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY

    def make_line_comment(self, text: str) -> str:
        return f"-- {text}"