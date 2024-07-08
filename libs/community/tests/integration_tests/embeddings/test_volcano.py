"""Test Bytedance Volcano Embedding."""

from langchain_community.embeddings import VolcanoEmbeddings


def test_embedding_documents() -> None:
    """Test embeddings for documents."""
    documents = ["foo", "bar"]
    embedding = VolcanoEmbeddings()  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1024


def test_embedding_query() -> None:
    """Test embeddings for query."""
    document = "foo bar"
    embedding = VolcanoEmbeddings()  # type: ignore[call-arg]
    output = embedding.embed_query(document)
    assert len(output) == 1024
