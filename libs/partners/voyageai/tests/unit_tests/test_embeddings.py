"""Test embedding model integration."""
import os

from langchain_voyageai.embeddings import VoyageAIEmbeddings
from langchain_core.embeddings import Embeddings

MODEL="voyage-2"


def test_initialization() -> None:
    """Test embedding model initialization."""
    emb = VoyageAIEmbeddings(voyage_api_key="NOT_A_VALID_KEY", model=MODEL)
    assert isinstance(emb, Embeddings)
