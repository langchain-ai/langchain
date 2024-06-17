"""Test embedding model integration."""

from langchain_mixedbreadai.embeddings import MixedbreadAIEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    MixedbreadAIEmbeddings(mxbai_api_key="test")
