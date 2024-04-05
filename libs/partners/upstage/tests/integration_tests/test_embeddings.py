"""Test Upstage embeddings."""
from langchain_upstage.embeddings import UpstageEmbeddings


def test_langchain_upstage_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = UpstageEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_upstage_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = UpstageEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
