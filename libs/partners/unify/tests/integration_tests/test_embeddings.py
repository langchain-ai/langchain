"""Test Unify embeddings."""
from langchain_unify.embeddings import UnifyEmbeddings


def test_langchain_unify_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = UnifyEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_unify_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = UnifyEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
