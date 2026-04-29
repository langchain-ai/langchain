"""Unit tests for `PerplexityEmbeddings`."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

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
    assert embeddings.pplx_api_key is not None
    assert embeddings.pplx_api_key.get_secret_value() == "test"
    assert embeddings.model == "pplx-embed-v1-4b"
    assert embeddings.client is not None
    assert embeddings.async_client is not None


def test_embeddings_custom_model() -> None:
    embeddings = PerplexityEmbeddings(pplx_api_key="test", model="custom-model")
    assert embeddings.model == "custom-model"


def test_api_key_alias() -> None:
    """`api_key=` should be accepted via populate_by_name alias."""
    embeddings = PerplexityEmbeddings(api_key="aliased")
    assert embeddings.pplx_api_key is not None
    assert embeddings.pplx_api_key.get_secret_value() == "aliased"


def test_api_key_accepts_secret_str() -> None:
    embeddings = PerplexityEmbeddings(pplx_api_key=SecretStr("typed"))
    assert embeddings.pplx_api_key is not None
    assert embeddings.pplx_api_key.get_secret_value() == "typed"


def test_lc_secrets() -> None:
    embeddings = PerplexityEmbeddings(pplx_api_key="test")
    assert embeddings.lc_secrets == {"pplx_api_key": "PPLX_API_KEY"}


def test_pplx_api_key_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("PPLX_API_KEY", "from_pplx_env")
    embeddings = PerplexityEmbeddings()
    assert embeddings.pplx_api_key is not None
    assert embeddings.pplx_api_key.get_secret_value() == "from_pplx_env"


def test_perplexity_api_key_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    monkeypatch.setenv("PERPLEXITY_API_KEY", "from_perp_env")
    embeddings = PerplexityEmbeddings()
    assert embeddings.pplx_api_key is not None
    assert embeddings.pplx_api_key.get_secret_value() == "from_perp_env"


def test_pplx_takes_precedence_over_perplexity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PPLX_API_KEY", "primary")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "secondary")
    embeddings = PerplexityEmbeddings()
    assert embeddings.pplx_api_key is not None
    assert embeddings.pplx_api_key.get_secret_value() == "primary"


def test_explicit_kwarg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PPLX_API_KEY", "from_env")
    embeddings = PerplexityEmbeddings(pplx_api_key="explicit")
    assert embeddings.pplx_api_key is not None
    assert embeddings.pplx_api_key.get_secret_value() == "explicit"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Perplexity API key not provided"):
        PerplexityEmbeddings()


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


def test_embed_documents_empty_short_circuits() -> None:
    mock_client = MagicMock()
    embeddings = PerplexityEmbeddings(pplx_api_key="test", client=mock_client)

    assert embeddings.embed_documents([]) == []
    mock_client.embeddings.create.assert_not_called()


def test_embed_documents_propagates_errors() -> None:
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = RuntimeError("boom")
    embeddings = PerplexityEmbeddings(pplx_api_key="test", client=mock_client)

    with pytest.raises(RuntimeError, match="boom"):
        embeddings.embed_documents(["x"])


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


async def test_aembed_documents_empty_short_circuits() -> None:
    mock_async_client = MagicMock()
    mock_async_client.embeddings.create = AsyncMock()
    embeddings = PerplexityEmbeddings(
        pplx_api_key="test", async_client=mock_async_client
    )

    assert await embeddings.aembed_documents([]) == []
    mock_async_client.embeddings.create.assert_not_awaited()


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
