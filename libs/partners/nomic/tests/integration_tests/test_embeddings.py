"""Test Nomic embeddings."""

from langchain_nomic.embeddings import NomicEmbeddings


def test_langchain_nomic_embedding_documents() -> None:
    """Test nomic embeddings."""
    documents = ["foo bar"]
    embedding = NomicEmbeddings(model="nomic-embed-text-v1")
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_nomic_embedding_query() -> None:
    """Test nomic embeddings."""
    document = "foo bar"
    embedding = NomicEmbeddings(model="nomic-embed-text-v1")
    output = embedding.embed_query(document)
    assert len(output) > 0


def test_langchain_nomic_embedding_dimensionality() -> None:
    """Test nomic embeddings."""
    documents = ["foo bar"]
    embedding = NomicEmbeddings(model="nomic-embed-text-v1.5", dimensionality=256)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 256
