"""Test Pinecone embeddings."""
from langchain_pinecone.embeddings import PineconeEmbeddings


def test_langchain_pinecone_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = PineconeEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_pinecone_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = PineconeEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
