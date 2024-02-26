"""Test embedding model integration."""


from langchain_voyageai.embeddings import VoyageAIEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    VoyageAIEmbeddings()
