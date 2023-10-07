from langchain.document_loaders.parsers.language.tree_sitter_segmenter import TreeSitterSegmenter

# TODO: Write the real query
CHUNK_QUERY = """
    [
        (class_specifier
            (field_declaration_list)) @chunk
    ]
""".strip()

class CPPSegmenter(TreeSitterSegmenter):
    """Code segmenter for C++."""

    def get_chunk_query(self) -> str:
        return CHUNK_QUERY
