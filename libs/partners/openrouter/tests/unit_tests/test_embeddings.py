"""Test embedding model integration."""

from typing import Any
from unittest.mock import Mock, patch

from langchain_openrouter.embeddings import OpenRouterEmbeddings

MODEL_NAME = "qwen/qwen3-embedding-8b"


def test_initialization() -> None:
    """Test embedding model initialization."""
    OpenRouterEmbeddings(model=MODEL_NAME)


@patch("langchain_openrouter.embeddings.openai.OpenAI")
def test_client_initialization(mock_openai_client: Any) -> None:
    """Test that OpenRouter client is properly initialized."""
    mock_client = Mock()
    mock_openai_client.return_value = mock_client
    mock_client.embeddings = Mock()

    embeddings = OpenRouterEmbeddings(model=MODEL_NAME)

    # Check that OpenAI client was initialized
    mock_openai_client.assert_called_once()
    assert embeddings._client is not None


@patch("langchain_openrouter.embeddings.openai.AsyncOpenAI")
def test_async_client_initialization(mock_async_client_class: Any) -> None:
    """Test that OpenRouter async client is properly initialized."""
    mock_client = Mock()
    mock_async_client_class.return_value = mock_client
    mock_client.embeddings = Mock()

    embeddings = OpenRouterEmbeddings(model=MODEL_NAME)

    # Check that AsyncOpenAI client was initialized
    mock_async_client_class.assert_called_once()
    assert embeddings._async_client is not None


def test_embed_documents() -> None:
    """Test that embed_documents works correctly."""
    with patch(
        "langchain_openrouter.embeddings.OpenRouterEmbeddings._embed_with_payload"
    ) as mock_embed:
        mock_embed.return_value = [0.1, 0.2, 0.3]

        embeddings = OpenRouterEmbeddings(model=MODEL_NAME)
        result = embeddings.embed_documents(["test text"])

        assert result == [[0.1, 0.2, 0.3]]
        mock_embed.assert_called_once_with("test text")


def test_embed_query() -> None:
    """Test that embed_query works correctly."""
    with patch(
        "langchain_openrouter.embeddings.OpenRouterEmbeddings._embed_with_payload"
    ) as mock_embed:
        mock_embed.return_value = [0.1, 0.2, 0.3]

        embeddings = OpenRouterEmbeddings(model=MODEL_NAME)
        result = embeddings.embed_query("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_embed.assert_called_once_with("test text")
