from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_perplexity import PerplexityEmbeddings


def _make_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = []
    for v in vectors:
        item = MagicMock()
        item.embedding = v
        response.data.append(item)
    return response


def test_embeddings_initialization() -> None:
    embeddings = PerplexityEmbeddings(pplx_api_key="test")
    assert embeddings.pplx_api_key.get_secret_value() == "test"
    assert embeddings.model == "pplx-embed-v1-4b"
    assert embeddings.client is not None
    assert embeddings.async_client is not None


def test_embeddings_custom_model() -> None:
    embeddings = PerplexityEmbeddings(pplx_api_key="test", model="custom-model")
    assert embeddings.model == "custom-model"


def test_embed_documents() -> None:
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_response(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    embeddings = PerplexityEmbeddings(pplx_api_key="test", client=mock_client)

    result = embeddings.embed_documents(["hello", "world"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_client.embeddings.create.assert_called_once_with(
        model="pplx-embed-v1-4b", input=["hello", "world"]
    )


def test_embed_query() -> None:
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_response([[0.7, 0.8, 0.9]])
    embeddings = PerplexityEmbeddings(pplx_api_key="test", client=mock_client)

    result = embeddings.embed_query("hello")

    assert result == [0.7, 0.8, 0.9]
    mock_client.embeddings.create.assert_called_once_with(
        model="pplx-embed-v1-4b", input=["hello"]
    )


def test_embed_documents_uses_custom_model() -> None:
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_response([[0.0]])
    embeddings = PerplexityEmbeddings(
        pplx_api_key="test", model="custom-model", client=mock_client
    )

    embeddings.embed_documents(["x"])

    mock_client.embeddings.create.assert_called_once_with(
        model="custom-model", input=["x"]
    )


@pytest.mark.asyncio
async def test_aembed_documents() -> None:
    mock_async_client = MagicMock()
    mock_async_client.embeddings.create = AsyncMock(
        return_value=_make_response([[0.1, 0.2], [0.3, 0.4]])
    )
    embeddings = PerplexityEmbeddings(
        pplx_api_key="test", async_client=mock_async_client
    )

    result = await embeddings.aembed_documents(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_async_client.embeddings.create.assert_awaited_once_with(
        model="pplx-embed-v1-4b", input=["a", "b"]
    )


@pytest.mark.asyncio
async def test_aembed_query() -> None:
    mock_async_client = MagicMock()
    mock_async_client.embeddings.create = AsyncMock(
        return_value=_make_response([[0.5, 0.6]])
    )
    embeddings = PerplexityEmbeddings(
        pplx_api_key="test", async_client=mock_async_client
    )

    result = await embeddings.aembed_query("hi")

    assert result == [0.5, 0.6]
    mock_async_client.embeddings.create.assert_awaited_once_with(
        model="pplx-embed-v1-4b", input=["hi"]
    )
