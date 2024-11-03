"""Test Fireworks embeddings."""

from langchain_fireworks.embeddings import FireworksEmbeddings


def test_langchain_fireworks_embedding_documents() -> None:
    """Test Fireworks hosted embeddings."""
    documents = ["foo bar"]
    embedding = FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_fireworks_embedding_query() -> None:
    """Test Fireworks hosted embeddings."""
    document = "foo bar"
    embedding = FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
    output = embedding.embed_query(document)
    assert len(output) > 0
