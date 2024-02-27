"""Test embedding model integration."""
import os

from langchain_voyageai.embeddings import VoyageAIEmbeddings
from langchain_core.embeddings import Embeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    emb = VoyageAIEmbeddings(voyage_api_key="NOT_A_VALID_KEY")
    assert isinstance(emb, Embeddings)


def test_initialization_api_key_in_environment() -> None:
    """Test embedding model initialization."""
    os.environ["VOYAGE_API_KEY"] = "NOT_A_VALID_KEY"
    emb = VoyageAIEmbeddings()
    assert isinstance(emb, Embeddings)
