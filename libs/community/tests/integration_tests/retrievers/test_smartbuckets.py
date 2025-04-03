"""Tests for the Raindrop retriever."""

from typing import Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document
from raindrop_retriever import RaindropRetriever


@pytest.fixture
def mock_raindrop_client() -> Generator[Mock, None, None]:
    """Create a mock Raindrop client."""
    with patch("raindrop_retriever.Raindrop") as mock:
        client = Mock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_async_raindrop_client() -> Generator[Mock, None, None]:
    """Create a mock AsyncRaindrop client."""
    with patch("raindrop_retriever.AsyncRaindrop") as mock:
        client = Mock()
        # Create an AsyncMock for the chunk_search.create method
        chunk_search = Mock()
        chunk_search.create = AsyncMock()
        client.chunk_search = chunk_search
        mock.return_value = client
        yield client


@pytest.fixture
def retriever(
    mock_raindrop_client: Mock,
    mock_async_raindrop_client: Mock,
) -> RaindropRetriever:
    """Create a retriever instance with mock clients."""
    return RaindropRetriever(api_key="test-key")


def test_retriever_initialization() -> None:
    """Test retriever initialization with API key."""
    with (
        patch("raindrop_retriever.Raindrop"),
        patch("raindrop_retriever.AsyncRaindrop"),
    ):
        retriever = RaindropRetriever(api_key="test-key")
        assert retriever.api_key == "test-key"
        assert retriever.client is not None
        assert retriever.async_client is not None


def test_retriever_initialization_with_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test retriever initialization with environment variable."""
    with (
        patch("raindrop_retriever.Raindrop"),
        patch("raindrop_retriever.AsyncRaindrop"),
    ):
        monkeypatch.setenv("RAINDROP_API_KEY", "env-key")
        retriever = RaindropRetriever()
        assert retriever.api_key == "env-key"
        assert retriever.client is not None
        assert retriever.async_client is not None


def test_retriever_initialization_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test retriever initialization without API key."""
    # Clear any existing environment variable
    monkeypatch.delenv("RAINDROP_API_KEY", raising=False)

    with pytest.raises(ValueError, match="No API key provided"):
        RaindropRetriever()


def test_invoke(retriever: RaindropRetriever, mock_raindrop_client: Mock) -> None:
    """Test the invoke method."""
    # Mock the chunk search response
    mock_result = Mock()
    mock_result.text = "Test content"
    mock_result.chunk_signature = "test-chunk"
    mock_result.payload_signature = "test-payload"
    mock_result.score = 0.95
    mock_result.type = "text/plain"
    mock_result.source = '{"bucket": "test-bucket", "path": "test.pdf"}'

    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_raindrop_client.chunk_search.create.return_value = mock_response

    # Get documents using invoke
    docs = retriever.invoke("test query")

    # Verify results
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "Test content"
    assert docs[0].metadata["chunk_signature"] == "test-chunk"
    assert docs[0].metadata["score"] == 0.95


@pytest.mark.asyncio
async def test_ainvoke(
    retriever: RaindropRetriever,
    mock_async_raindrop_client: Mock,
) -> None:
    """Test the ainvoke method."""
    # Mock the chunk search response
    mock_result = Mock()
    mock_result.text = "Test content"
    mock_result.chunk_signature = "test-chunk"
    mock_result.payload_signature = "test-payload"
    mock_result.score = 0.95
    mock_result.type = "text/plain"
    mock_result.source = '{"bucket": "test-bucket", "path": "test.pdf"}'

    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_async_raindrop_client.chunk_search.create.return_value = mock_response

    # Get documents using ainvoke
    docs = await retriever.ainvoke("test query")

    # Verify results
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "Test content"
    assert docs[0].metadata["chunk_signature"] == "test-chunk"
    assert docs[0].metadata["score"] == 0.95


def test_error_handling(
    retriever: RaindropRetriever,
    mock_raindrop_client: Mock,
) -> None:
    """Test error handling in document retrieval."""
    mock_raindrop_client.chunk_search.create.side_effect = Exception("API Error")

    with pytest.raises(Exception, match="API Error"):
        retriever.invoke("test query")
