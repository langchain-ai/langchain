"""Test embedding model integration."""

from langchain_ollama.embeddings import OllamaEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    OllamaEmbeddings(model="llama3", keep_alive=1)
