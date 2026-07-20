"""Test OpenRouterReranker."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from pydantic import SecretStr

from langchain_openrouter.rerank import OpenRouterRerank


def test_initialization() -> None:
    """Test reranker initialization."""
    reranker = OpenRouterRerank(
        model_name="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        top_n=2,
        api_key=SecretStr("test-key"),
    )
    assert reranker.model == "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
    assert reranker.top_n == 2


def test_serialize_documents() -> None:
    """Test document serialization."""
    docs = [
        Document(page_content="Test string"),
        Document(
            page_content="Test image", metadata={"image_url": "https://example.com"}
        ),
    ]

    serialized = OpenRouterRerank._serialize_documents(docs)

    assert len(serialized) == 2
    assert serialized[0] == "Test string"
    assert serialized[1] == {"text": "Test image", "image": "https://example.com"}


def test_app_url_passed_to_client() -> None:
    """Test that app_url is passed as HTTP-Referer header via httpx clients."""
    with patch("openrouter.OpenRouter") as mock_cls:
        mock_cls.return_value = MagicMock()
        OpenRouterRerank(
            model_name="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
            api_key=SecretStr("test-key"),
            app_url="https://myapp.com",
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["client"].headers["HTTP-Referer"] == "https://myapp.com"


def test_app_title_passed_to_client() -> None:
    """Test that app_title is passed as X-Title header via httpx clients."""
    with patch("openrouter.OpenRouter") as mock_cls:
        mock_cls.return_value = MagicMock()
        OpenRouterRerank(
            model_name="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
            api_key=SecretStr("test-key"),
            app_title="My App",
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["client"].headers["X-Title"] == "My App"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that missing API key raises an error."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY must be set"):
        OpenRouterRerank(model_name="nvidia/llama-nemotron-rerank-vl-1b-v2:free")


def test_compress_documents() -> None:
    """Test compress_documents endpoint interaction."""
    mock_response = MagicMock()
    mock_result_1 = MagicMock()
    mock_result_1.index = 1
    mock_result_1.relevance_score = 0.9

    mock_result_2 = MagicMock()
    mock_result_2.index = 0
    mock_result_2.relevance_score = 0.4

    mock_response.results = [mock_result_1, mock_result_2]

    reranker = OpenRouterRerank(
        model_name="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        api_key=SecretStr("test-key"),
    )

    reranker.client = MagicMock()
    reranker.client.rerank.rerank.return_value = mock_response

    docs = [
        Document(page_content="Bad match"),
        Document(page_content="Good match"),
    ]

    compressed_docs = reranker.compress_documents(documents=docs, query="match")

    assert len(compressed_docs) == 2
    assert compressed_docs[0].page_content == "Good match"
    assert compressed_docs[0].metadata["relevance_score"] == 0.9

    assert compressed_docs[1].page_content == "Bad match"
    assert compressed_docs[1].metadata["relevance_score"] == 0.4

    reranker.client.rerank.rerank.assert_called_once_with(
        model="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        documents=["Bad match", "Good match"],
        query="match",
        top_n=3,
    )
