"""Test embedding model integration."""

from typing import Any
from unittest.mock import Mock, patch

from langchain_ollama.embeddings import OllamaEmbeddings

MODEL_NAME = "llama3.1"


def test_initialization() -> None:
    """Test embedding model initialization."""
    OllamaEmbeddings(model=MODEL_NAME, keep_alive=1)


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


@patch("langchain_ollama.embeddings.Client")
def test_embed_documents_passes_options(mock_client_class: Any) -> None:
    """Test that embed_documents method passes options including num_gpu."""
    # Create a mock client instance
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    # Mock the embed method response
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

    # Create embeddings with num_gpu parameter
    embeddings = OllamaEmbeddings(model=MODEL_NAME, num_gpu=4, temperature=0.5)

    # Call embed_documents
    result = embeddings.embed_documents(["test text"])

    # Verify the result
    assert result == [[0.1, 0.2, 0.3]]

    # Check that embed was called with correct arguments
    mock_client.embed.assert_called_once()
    call_args = mock_client.embed.call_args

    # Verify the keyword arguments
    assert "options" in call_args.kwargs
    assert "keep_alive" in call_args.kwargs

    # Verify options contain num_gpu and temperature
    options = call_args.kwargs["options"]
    assert options["num_gpu"] == 4
    assert options["temperature"] == 0.5
