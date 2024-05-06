"""Test embedding model integration."""

from langchain_fireworks.embeddings import FireworksEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
