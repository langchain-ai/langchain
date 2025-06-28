"""Test embedding model integration."""

from langchain_ollama.embeddings import OllamaEmbeddings

MODEL_NAME = "llama3.1"


def test_initialization() -> None:
    """Test embedding model initialization."""
    OllamaEmbeddings(model=MODEL_NAME, keep_alive=1)
