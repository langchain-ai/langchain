"""Unit tests for OpenRouter embeddings."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from langchain_openrouter.embeddings import OpenRouterEmbeddings


def test_openrouter_embeddings_embed_documents() -> None:
    embeddings = OpenRouterEmbeddings(
        openrouter_api_key=SecretStr("test-key"),
        openrouter_api_base="https://openrouter.ai/api/v1",
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4]},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("langchain_openrouter.embeddings.httpx.Client") as client_cls:
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = mock_response
        result = embeddings.embed_documents(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    client.post.assert_called_once()
    assert client.post.call_args.kwargs["json"]["model"] == "openai/text-embedding-3-small"


def test_openrouter_embeddings_requires_api_key() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterEmbeddings(openrouter_api_key=None)
