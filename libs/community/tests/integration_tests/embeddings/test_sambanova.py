"""Test SambaNova Embeddings."""

from langchain_community.embeddings.sambanova import (
    SambaStudioEmbeddings,
)


def test_embedding_documents() -> None:
    """Test embeddings for documents."""
    documents = ["foo", "bar"]
    embedding = SambaStudioEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1024


def test_embedding_query() -> None:
    """Test embeddings for query."""
    document = "foo bar"
    embedding = SambaStudioEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1024
