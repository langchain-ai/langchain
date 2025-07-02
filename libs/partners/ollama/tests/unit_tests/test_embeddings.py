"""Test embedding model integration."""

from typing import Any
from unittest.mock import patch

from langchain_ollama.embeddings import OllamaEmbeddings

MODEL_NAME = "llama3.1"


def test_initialization() -> None:
    """Test embedding model initialization."""
    OllamaEmbeddings(model="llama3", keep_alive=1)


@patch("langchain_ollama.embeddings.validate_model")
def test_validate_model_on_init(mock_validate_model: Any) -> None:
    """Test that the model is validated on initialization when requested."""
    # Test that validate_model is called when validate_model_on_init=True
    OllamaEmbeddings(model=MODEL_NAME, validate_model_on_init=True)
    mock_validate_model.assert_called_once()
    mock_validate_model.reset_mock()

    # Test that validate_model is NOT called when validate_model_on_init=False
    OllamaEmbeddings(model=MODEL_NAME, validate_model_on_init=False)
    mock_validate_model.assert_not_called()

    # Test that validate_model is NOT called by default
    OllamaEmbeddings(model=MODEL_NAME)
    mock_validate_model.assert_not_called()
