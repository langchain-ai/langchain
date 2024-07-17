from langchain_core.embeddings import FakeEmbeddings

from langchain_experimental.text_splitter import SemanticChunker


def test_semantic_chunker() -> None:
    """Test the split_text method of SemanticChunker."""
    text = "This is a sentence. This is another sentence."
    expected_chunks = [
        "This ",
        "is a ",
        "sente",
        "nce.",
        "This ",
        "is an",
        "other",
        " sent",
        "ence.",
    ]

    embeddings = FakeEmbeddings(size=1)
    chunker = SemanticChunker(embeddings=embeddings, buffer_size=1, max_chunk_size=5)

    result_chunks = chunker.split_text(text)
    assert result_chunks == expected_chunks
