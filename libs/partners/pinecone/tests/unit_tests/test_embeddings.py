"""Test embedding model integration."""


from langchain_pinecone.embeddings import PineconeEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    PineconeEmbeddings()
