from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_perplexity import PerplexityEmbeddings


def _make_embedding_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    return response


@patch("langchain_perplexity.embeddings.Perplexity")
@patch("langchain_perplexity.embeddings.AsyncPerplexity")
def test_embeddings_model_param(mock_async: MagicMock, mock_sync: MagicMock) -> None:
    embeddings = PerplexityEmbeddings(model="pplx-embed-context-v1", pplx_api_key="test")
    assert embeddings.model == "pplx-embed-context-v1"


@patch("langchain_perplexity.embeddings.Perplexity")
@patch("langchain_perplexity.embeddings.AsyncPerplexity")
def test_embeddings_secrets_not_exposed(
    mock_async: MagicMock, mock_sync: MagicMock
) -> None:
    embeddings = PerplexityEmbeddings(pplx_api_key="supersecret")
    assert "supersecret" not in str(embeddings)


@patch("langchain_perplexity.embeddings.Perplexity")
@patch("langchain_perplexity.embeddings.AsyncPerplexity")
def test_embed_documents(mock_async: MagicMock, mock_sync: MagicMock) -> None:
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_embedding_response(vectors)
    mock_sync.return_value = mock_client

    embeddings = PerplexityEmbeddings(pplx_api_key="test")
    result = embeddings.embed_documents(["hello", "world"])

    mock_client.embeddings.create.assert_called_once_with(
        input=["hello", "world"], model="pplx-embed-v1"
    )
    assert result == vectors


@patch("langchain_perplexity.embeddings.Perplexity")
@patch("langchain_perplexity.embeddings.AsyncPerplexity")
def test_embed_query(mock_async: MagicMock, mock_sync: MagicMock) -> None:
    vector = [0.1, 0.2, 0.3]
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_embedding_response([vector])
    mock_sync.return_value = mock_client

    embeddings = PerplexityEmbeddings(pplx_api_key="test")
    result = embeddings.embed_query("what is perplexity?")

    mock_client.embeddings.create.assert_called_once_with(
        input=["what is perplexity?"], model="pplx-embed-v1"
    )
    assert result == vector


@pytest.mark.asyncio
@patch("langchain_perplexity.embeddings.Perplexity")
@patch("langchain_perplexity.embeddings.AsyncPerplexity")
async def test_aembed_documents(mock_async: MagicMock, mock_sync: MagicMock) -> None:
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = _make_embedding_response(vectors)
    mock_async.return_value = mock_client

    embeddings = PerplexityEmbeddings(pplx_api_key="test")
    result = await embeddings.aembed_documents(["foo", "bar"])

    mock_client.embeddings.create.assert_called_once_with(
        input=["foo", "bar"], model="pplx-embed-v1"
    )
    assert result == vectors


@pytest.mark.asyncio
@patch("langchain_perplexity.embeddings.Perplexity")
@patch("langchain_perplexity.embeddings.AsyncPerplexity")
async def test_aembed_query(mock_async: MagicMock, mock_sync: MagicMock) -> None:
    vector = [0.7, 0.8, 0.9]
    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = _make_embedding_response([vector])
    mock_async.return_value = mock_client

    embeddings = PerplexityEmbeddings(pplx_api_key="test")
    result = await embeddings.aembed_query("async query")

    assert result == vector
