"""Unit tests for FireworksRerank."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from langchain_core.documents import Document

from langchain_fireworks.rerank import FireworksRerank


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock OpenAI client."""
    client = MagicMock()
    return client


@pytest.fixture
@patch("langchain_fireworks.rerank.FireworksRerank.validate_environment")
def mock_rerank(mock_validate: MagicMock) -> FireworksRerank:
    """Create a FireworksRerank instance with mocked client."""
    reranker = FireworksRerank(
        model="accounts/fireworks/models/reranker",
        top_n=3,
        fireworks_api_key="test-api-key",
    )
    return reranker


def test_rerank_initialization() -> None:
    """Test FireworksRerank initialization with default parameters."""
    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"):
        reranker = FireworksRerank(
            model="custom-model",
            top_n=5,
            fireworks_api_key="test-key",
        )

    assert reranker.model == "custom-model"
    assert reranker.top_n == 5
    assert reranker.fireworks_api_key.get_secret_value() == "test-key"


def test_rerank_initialization_defaults() -> None:
    """Test FireworksRerank initialization with default parameters."""
    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"):
        reranker = FireworksRerank(fireworks_api_key="test-key")

    assert reranker.model == "accounts/fireworks/models/reranker"
    assert reranker.top_n == 3


def test_rerank_empty_documents() -> None:
    """Test rerank with empty documents returns empty list."""
    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"):
        reranker = FireworksRerank(fireworks_api_key="test-key")

    # Mock the client
    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.json.return_value = {"data": []}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch.object(FireworksRerank, "validate_environment"), \
         patch("langchain_fireworks.rerank.OpenAI") as mock_openai:
        mock_openai.return_value = mock_response
        reranker = FireworksRerank(fireworks_api_key="test-key")
        reranker.client = mock_client

        result = reranker.rerank([], "test query")

        assert result == []


def test_rerank_with_documents() -> None:
    """Test rerank with documents returns proper format."""
    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"):
        reranker = FireworksRerank(fireworks_api_key="test-key")

    # Mock the client response
    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.json.return_value = {
        "data": [
            {"index": 1, "score": 0.95},
            {"index": 0, "score": 0.85},
        ]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"), \
         patch("langchain_fireworks.rerank.OpenAI") as mock_openai:
        mock_openai.return_value = mock_response
        reranker = FireworksRerank(fireworks_api_key="test-key")
        reranker.client = mock_client

        documents = ["doc1 content", "doc2 content", "doc3 content"]
        result = reranker.rerank(documents, "test query")

        assert len(result) == 2
        assert result[0]["index"] == 1
        assert result[0]["relevance_score"] == 0.95
        assert result[1]["index"] == 0
        assert result[1]["relevance_score"] == 0.85


def test_compress_documents() -> None:
    """Test compress_documents with reranked results."""
    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"):
        reranker = FireworksRerank(fireworks_api_key="test-key")

    # Mock the client response for rerank
    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.json.return_value = {
        "data": [
            {"index": 1, "score": 0.95},
            {"index": 0, "score": 0.85},
        ]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"), \
         patch("langchain_fireworks.rerank.OpenAI") as mock_openai:
        mock_openai.return_value = mock_response
        reranker = FireworksRerank(fireworks_api_key="test-key")
        reranker.client = mock_client

        documents = [
            Document(page_content="doc1", metadata={"source": "doc1"}),
            Document(page_content="doc2", metadata={"source": "doc2"}),
            Document(page_content="doc3", metadata={"source": "doc3"}),
        ]

        result = reranker.compress_documents(documents, "test query")

        assert len(result) == 2
        assert isinstance(result[0], Document)
        assert result[0].metadata["relevance_score"] == 0.95
        assert result[1].metadata["relevance_score"] == 0.85
        # Original documents should not be modified
        assert "relevance_score" not in documents[0].metadata


def test_compress_documents_preserves_metadata() -> None:
    """Test that compress_documents preserves original metadata."""
    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"):
        reranker = FireworksRerank(fireworks_api_key="test-key")

    # Mock the client response for rerank
    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.json.return_value = {
        "data": [
            {"index": 0, "score": 0.9},
        ]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"), \
         patch("langchain_fireworks.rerank.OpenAI") as mock_openai:
        mock_openai.return_value = mock_response
        reranker = FireworksRerank(fireworks_api_key="test-key")
        reranker.client = mock_client

        original_metadata = {"source": "test.txt", "page": 1}
        documents = [
            Document(page_content="test", metadata=original_metadata),
        ]

        result = reranker.compress_documents(documents, "test query")

        assert len(result) == 1
        # Check that original metadata is preserved
        assert result[0].metadata["source"] == "test.txt"
        assert result[0].metadata["page"] == 1
        # Check that relevance_score is added
        assert result[0].metadata["relevance_score"] == 0.9


def test_rerank_top_n_parameter() -> None:
    """Test rerank respects top_n parameter."""
    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"):
        reranker = FireworksRerank(fireworks_api_key="test-key", top_n=2)

    mock_response = MagicMock()
    mock_response.is_success = True
    # Mock API returns only top_n results (2)
    mock_response.json.return_value = {
        "data": [
            {"index": 0, "score": 0.9},
            {"index": 1, "score": 0.8},
        ]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"), \
         patch("langchain_fireworks.rerank.OpenAI") as mock_openai:
        mock_openai.return_value = mock_response
        reranker = FireworksRerank(fireworks_api_key="test-key", top_n=2)
        reranker.client = mock_client

        documents = ["doc1", "doc2", "doc3", "doc4"]
        result = reranker.rerank(documents, "test query", top_n=2)

        # Should only return top_n results
        assert len(result) == 2
        # Check that top_n was passed to API
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["top_n"] == 2


def test_compress_documents_async() -> None:
    """Test async compress documents."""
    import asyncio

    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"):
        reranker = FireworksRerank(fireworks_api_key="test-key")

    # Mock the client response for rerank
    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.json.return_value = {
        "data": [
            {"index": 0, "score": 0.9},
        ]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"), \
         patch("langchain_fireworks.rerank.OpenAI") as mock_openai:
        mock_openai.return_value = mock_response
        reranker = FireworksRerank(fireworks_api_key="test-key")
        reranker.client = mock_client

        documents = [
            Document(page_content="doc1", metadata={"source": "doc1"}),
        ]

        async def test_async():
            result = await reranker.acompress_documents(documents, "test query")
            assert len(result) == 1
            assert isinstance(result[0], Document)
            assert result[0].metadata["relevance_score"] == 0.9

        asyncio.run(test_async())


def test_rerank_model_override() -> None:
    """Test rerank allows model override."""
    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"):
        reranker = FireworksRerank(fireworks_api_key="test-key")

    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.json.return_value = {"data": []}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"), \
         patch("langchain_fireworks.rerank.OpenAI") as mock_openai:
        mock_openai.return_value = mock_response
        reranker = FireworksRerank(fireworks_api_key="test-key")
        reranker.client = mock_client

        reranker.rerank(["doc1"], "query", model="custom-model")

        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["model"] == "custom-model"


def test_rerank_with_document_objects() -> None:
    """Test rerank accepts Document objects."""
    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"):
        reranker = FireworksRerank(fireworks_api_key="test-key")

    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.json.return_value = {"data": []}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("langchain_fireworks.rerank.FireworksRerank.validate_environment"), \
         patch("langchain_fireworks.rerank.OpenAI") as mock_openai:
        mock_openai.return_value = mock_response
        reranker = FireworksRerank(fireworks_api_key="test-key")
        reranker.client = mock_client

        documents = [
            Document(page_content="doc1 content"),
            "plain string doc",
            {"content": "dict doc"},
        ]
        result = reranker.rerank(documents, "query")

        # Should convert all to content strings
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["documents"] == [
            "doc1 content",
            "plain string doc",
            {"content": "dict doc"},
        ]