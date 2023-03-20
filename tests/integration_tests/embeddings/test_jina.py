"""Test jina embeddings."""
from langchain.embeddings.jina import JinaEmbeddings


def test_cohere_embedding_documents() -> None:
    """Test jina embeddings for documents."""
    documents = ["foo bar"]
    embedding = JinaEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 2048


def test_cohere_embedding_query() -> None:
    """Test jina embeddings for query."""
    document = "foo bar"
    embedding = JinaEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 2048
