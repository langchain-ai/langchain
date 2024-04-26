"""Test Prompty embeddings."""
from langchain_prompty.embeddings import PromptyEmbeddings


def test_langchain_prompty_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = PromptyEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_prompty_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = PromptyEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
