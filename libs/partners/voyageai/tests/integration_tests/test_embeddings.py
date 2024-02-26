"""Test VoyageAI embeddings."""
from langchain_voyageai.embeddings import VoyageAIEmbeddings


def test_langchain_voyageai_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = VoyageAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_voyageai_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = VoyageAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
