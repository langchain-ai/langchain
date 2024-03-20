"""Test Cohere embeddings."""
from langchain_cohere import CohereEmbeddings


def test_langchain_cohere_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = CohereEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_cohere_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = CohereEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
