"""Test Together embeddings."""
from langchain_together.embeddings import TogetherEmbeddings


def test_langchain_together_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = TogetherEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_together_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = TogetherEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
