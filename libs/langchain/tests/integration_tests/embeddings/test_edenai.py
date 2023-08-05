"""Test edenai embeddings."""

from langchain.embeddings.edenai import EdenAiEmbeddings


def test_edenai_embedding_documents() -> None:
    """Test edenai embeddings with openai."""
    documents = ["foo bar", "test text"]
    embedding = EdenAiEmbeddings(provider="openai")
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536


def test_edenai_embedding_query() -> None:
    """Test eden ai embeddings with google."""
    document = "foo bar"
    embedding = EdenAiEmbeddings(provider="google")
    output = embedding.embed_query(document)
    assert len(output) == 768
