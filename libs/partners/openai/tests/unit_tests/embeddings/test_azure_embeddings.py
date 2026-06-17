import os
from unittest import mock

import httpx

from langchain_openai import AzureOpenAIEmbeddings


def test_initialize_azure_openai() -> None:
    embeddings = AzureOpenAIEmbeddings(  # type: ignore[call-arg]
        model="text-embedding-large",
        api_key="xyz",  # type: ignore[arg-type]
        azure_endpoint="my-base-url",
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
    )
    assert embeddings.model == "text-embedding-large"


def test_initialize_azure_openai_with_base_set() -> None:
    with mock.patch.dict(os.environ, {"OPENAI_API_BASE": "https://api.openai.com"}):
        embeddings = AzureOpenAIEmbeddings(  # type: ignore[call-arg, call-arg]
            model="text-embedding-large",
            api_key="xyz",  # type: ignore[arg-type]
            azure_endpoint="my-base-url",
            azure_deployment="35-turbo-dev",
            openai_api_version="2023-05-15",
            openai_api_base=None,
        )
        assert embeddings.model == "text-embedding-large"


@mock.patch("langchain_openai.embeddings.azure._get_default_httpx_client")
@mock.patch("langchain_openai.embeddings.azure._get_default_async_httpx_client")
def test_azure_embeddings_uses_default_httpx_client(
    mock_async: mock.MagicMock, mock_sync: mock.MagicMock
) -> None:
    """AzureOpenAIEmbeddings should use the cached default httpx client when none is provided."""
    mock_sync.return_value = httpx.Client()
    mock_async.return_value = httpx.AsyncClient()
    embeddings = AzureOpenAIEmbeddings(  # type: ignore[call-arg]
        api_key="test",  # type: ignore[arg-type]
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="text-embedding-3-large",
        openai_api_version="2024-02-01",
    )
    mock_sync.assert_called_once_with(
        embeddings.openai_api_base, embeddings.request_timeout
    )
    mock_async.assert_called_once_with(
        embeddings.openai_api_base, embeddings.request_timeout
    )


def test_azure_embeddings_custom_http_client_not_overridden() -> None:
    """A user-provided http_client must not be replaced by the default."""
    custom_client = httpx.Client()
    custom_async_client = httpx.AsyncClient()

    with mock.patch(
        "langchain_openai.embeddings.azure._get_default_httpx_client"
    ) as mock_sync, mock.patch(
        "langchain_openai.embeddings.azure._get_default_async_httpx_client"
    ) as mock_async:
        AzureOpenAIEmbeddings(  # type: ignore[call-arg]
            api_key="test",  # type: ignore[arg-type]
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="text-embedding-3-large",
            openai_api_version="2024-02-01",
            http_client=custom_client,
            http_async_client=custom_async_client,
        )
        mock_sync.assert_not_called()
        mock_async.assert_not_called()


def test_azure_embeddings_httpx_client_is_reused() -> None:
    """Two instances with identical config should share the same cached httpx client."""
    from langchain_openai.chat_models._client_utils import _get_default_httpx_client

    emb1 = AzureOpenAIEmbeddings(  # type: ignore[call-arg]
        api_key="test",  # type: ignore[arg-type]
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="text-embedding-3-large",
        openai_api_version="2024-02-01",
    )
    emb2 = AzureOpenAIEmbeddings(  # type: ignore[call-arg]
        api_key="test",  # type: ignore[arg-type]
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="text-embedding-3-large",
        openai_api_version="2024-02-01",
    )
    client1 = _get_default_httpx_client(emb1.openai_api_base, emb1.request_timeout)
    client2 = _get_default_httpx_client(emb2.openai_api_base, emb2.request_timeout)
    assert client1 is client2
