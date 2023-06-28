"""Test Google PaLM embeddings.

Note: This test must be run with the GOOGLE_API_KEY environment variable set to a
      valid API key.
"""
from langchain.embeddings.google_palm import GooglePalmEmbeddings


def test_google_palm_embedding_documents() -> None:
    """Test Google PaLM embeddings."""
    documents = ["foo bar"]
    embedding = GooglePalmEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_google_palm_embedding_documents_multiple() -> None:
    """Test Google PaLM embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = GooglePalmEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 768
    assert len(output[1]) == 768
    assert len(output[2]) == 768


def test_google_palm_embedding_query() -> None:
    """Test Google PaLM embeddings."""
    document = "foo bar"
    embedding = GooglePalmEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 768
