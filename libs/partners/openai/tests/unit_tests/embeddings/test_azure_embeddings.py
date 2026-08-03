import os
from unittest import mock

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


def test_azure_client_caching() -> None:
    """Test that the underlying httpx client is cached and reused."""
    embeddings1 = AzureOpenAIEmbeddings(  # type: ignore[call-arg]
        model="text-embedding-large",
        api_key="xyz",  # type: ignore[arg-type]
        azure_endpoint="my-base-url",
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
    )
    embeddings2 = AzureOpenAIEmbeddings(  # type: ignore[call-arg]
        model="text-embedding-large",
        api_key="xyz",  # type: ignore[arg-type]
        azure_endpoint="my-base-url",
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
    )
    assert embeddings1.client._client._client is embeddings2.client._client._client
    assert (
        embeddings1.async_client._client._client
        is embeddings2.async_client._client._client
    )

    embeddings3 = AzureOpenAIEmbeddings(  # type: ignore[call-arg]
        model="text-embedding-large",
        api_key="xyz",  # type: ignore[arg-type]
        azure_endpoint="other-base-url",
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
    )
    assert embeddings1.client._client._client is not embeddings3.client._client._client


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
