"""Test embedding model integration."""
import os

from langchain_voyageai.embeddings import VoyageAIEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    VoyageAIEmbeddings(voyage_api_key="NOT_A_VALID_KEY")


def test_initialization_api_key_in_environment() -> None:
    """Test embedding model initialization."""
    os.environ["VOYAGE_API_KEY"] = "NOT_A_VALID_KEY"
    VoyageAIEmbeddings()
