"""Test embedding model integration."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from langchain_ollama.embeddings import OllamaEmbeddings

MODEL_NAME = "llama3.1"


def test_initialization() -> None:
    """Test embedding model initialization."""
    OllamaEmbeddings(model=MODEL_NAME, keep_alive=1)


@patch("langchain_ollama.embeddings.validate_model")
def test_validate_model_on_init(mock_validate_model: Any) -> None:
    """Test that the model is validated on initialization when requested."""
    OllamaEmbeddings(model=MODEL_NAME, validate_model_on_init=True)
    mock_validate_model.assert_called_once()
    mock_validate_model.reset_mock()

    OllamaEmbeddings(model=MODEL_NAME, validate_model_on_init=False)
    mock_validate_model.assert_not_called()
    OllamaEmbeddings(model=MODEL_NAME)
    mock_validate_model.assert_not_called()


@patch("langchain_ollama.embeddings.Client")
def test_embed_documents_passes_options(mock_client_class: Any) -> None:
    """Test that `embed_documents()` passes options, including `num_gpu`."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

    embeddings = OllamaEmbeddings(model=MODEL_NAME, num_gpu=4, temperature=0.5)
    result = embeddings.embed_documents(["test text"])

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


def test_embed_documents_raises_when_client_none() -> None:
    """Test that embed_documents raises RuntimeError when client is None."""
    with patch("langchain_ollama.embeddings.Client") as mock_client_class:
        mock_client_class.return_value = MagicMock()
        embeddings = OllamaEmbeddings(model="test-model")
        embeddings._client = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="sync client is not initialized"):
            embeddings.embed_documents(["test"])


async def test_aembed_documents_raises_when_client_none() -> None:
    """Test that aembed_documents raises RuntimeError when async client is None."""
    with patch("langchain_ollama.embeddings.AsyncClient") as mock_client_class:
        mock_client_class.return_value = MagicMock()
        embeddings = OllamaEmbeddings(model="test-model")
        embeddings._async_client = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="async client is not initialized"):
            await embeddings.aembed_documents(["test"])
