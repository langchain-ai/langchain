"""Test Awa Embedding"""
from langchain.embeddings.awa import AwaEmbeddings


def test_awa_embedding_documents() -> None:
    """Test Awa embeddings for documents."""
    documents = ["foo bar", "test document"]
    embedding = AwaEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 768


def test_awa_embedding_query() -> None:
    """Test Awa embeddings for query."""
    document = "foo bar"
    embedding = AwaEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 768
