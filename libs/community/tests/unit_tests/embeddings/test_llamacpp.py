from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings


@pytest.fixture
def mock_llama_client() -> Generator[MagicMock, None, None]:
    with patch(
        "langchain_community.embeddings.llamacpp.LlamaCppEmbeddings"
    ) as MockLlama:
        mock_client = MagicMock()
        MockLlama.return_value = mock_client
        yield mock_client


def test_initialization(mock_llama_client: MagicMock) -> None:
    embeddings = LlamaCppEmbeddings(client=mock_llama_client)  # type: ignore[call-arg]
    assert embeddings.client is not None


def test_embed_documents(mock_llama_client: MagicMock) -> None:
    mock_llama_client.create_embedding.return_value = {
        "data": [{"embedding": [[0.1, 0.2, 0.3]]}, {"embedding": [[0.4, 0.5, 0.6]]}]
    }
    embeddings = LlamaCppEmbeddings(client=mock_llama_client)  # type: ignore[call-arg]
    texts = ["Hello world", "Test document"]
    result = embeddings.embed_documents(texts)
    expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert result == expected


def test_embed_query(mock_llama_client: MagicMock) -> None:
    mock_llama_client.embed.return_value = [[0.1, 0.2, 0.3]]
    embeddings = LlamaCppEmbeddings(client=mock_llama_client)  # type: ignore[call-arg]
    result = embeddings.embed_query("Sample query")
    expected = [0.1, 0.2, 0.3]
    assert result == expected
