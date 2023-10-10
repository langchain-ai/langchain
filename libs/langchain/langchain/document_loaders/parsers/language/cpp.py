from langchain.document_loaders.parsers.language.tree_sitter_segmenter import TreeSitterSegmenter

# TODO: Write the real query
CHUNK_QUERY = """
    [
        (class_specifier
            body: (field_declaration_list)) @class
        (function_definition) @function
    ]
""".strip()

class CPPSegmenter(TreeSitterSegmenter):
    """Code segmenter for C++."""

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY

    def make_line_comment(self, text: str) -> str:
        return f"// {text}"
