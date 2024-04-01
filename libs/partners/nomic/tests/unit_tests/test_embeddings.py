"""Test embedding model integration."""


from langchain_nomic.embeddings import NomicEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    NomicEmbeddings(model="nomic-embed-text-v1")
