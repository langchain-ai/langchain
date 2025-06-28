"""Test embedding model integration."""

import pytest

from langchain_ollama.embeddings import OllamaEmbeddings


@pytest.mark.enable_socket
def test_initialization() -> None:
    """Test embedding model initialization."""
    OllamaEmbeddings(model="llama3", keep_alive=1)
