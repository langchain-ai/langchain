"""Test Airbyte embeddings."""

from langchain_airbyte import AirbyteLoader


def test_langchain_airbyte_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = AirbyteEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_airbyte_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = AirbyteEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
