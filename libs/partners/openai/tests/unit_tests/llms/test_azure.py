from typing import Any
from unittest import mock

import httpx

from langchain_openai import AzureOpenAI


def test_azure_model_param(monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    llm = AzureOpenAI(
        openai_api_key="secret-api-key",  # type: ignore[call-arg]
        azure_endpoint="endpoint",
        api_version="version",
        azure_deployment="gpt-35-turbo-instruct",
    )

    # Test standard tracing params
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "azure",
        "ls_model_type": "llm",
        "ls_model_name": "gpt-35-turbo-instruct",
        "ls_temperature": 0.7,
        "ls_max_tokens": 256,
    }


@mock.patch("langchain_openai.llms.azure._get_default_httpx_client")
@mock.patch("langchain_openai.llms.azure._get_default_async_httpx_client")
def test_azure_llm_uses_default_httpx_client(
    mock_async: mock.MagicMock, mock_sync: mock.MagicMock
) -> None:
    """AzureOpenAI (LLM) should use the cached default httpx client when none is provided."""
    mock_sync.return_value = httpx.Client()
    mock_async.return_value = httpx.AsyncClient()
    llm = AzureOpenAI(
        openai_api_key="test",  # type: ignore[call-arg]
        azure_endpoint="https://test.openai.azure.com/",
        api_version="2024-02-01",
        azure_deployment="gpt-35-turbo-instruct",
    )
    mock_sync.assert_called_once_with(llm.openai_api_base, llm.request_timeout)
    mock_async.assert_called_once_with(llm.openai_api_base, llm.request_timeout)


def test_azure_llm_custom_http_client_not_overridden() -> None:
    """A user-provided http_client must not be replaced by the default."""
    custom_client = httpx.Client()
    custom_async_client = httpx.AsyncClient()

    with mock.patch(
        "langchain_openai.llms.azure._get_default_httpx_client"
    ) as mock_sync, mock.patch(
        "langchain_openai.llms.azure._get_default_async_httpx_client"
    ) as mock_async:
        AzureOpenAI(
            openai_api_key="test",  # type: ignore[call-arg]
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-02-01",
            azure_deployment="gpt-35-turbo-instruct",
            http_client=custom_client,
            http_async_client=custom_async_client,
        )
        mock_sync.assert_not_called()
        mock_async.assert_not_called()


def test_azure_llm_httpx_client_is_reused() -> None:
    """Two instances with identical config should share the same cached httpx client."""
    from langchain_openai.chat_models._client_utils import _get_default_httpx_client

    llm1 = AzureOpenAI(
        openai_api_key="test",  # type: ignore[call-arg]
        azure_endpoint="https://test.openai.azure.com/",
        api_version="2024-02-01",
        azure_deployment="gpt-35-turbo-instruct",
    )
    llm2 = AzureOpenAI(
        openai_api_key="test",  # type: ignore[call-arg]
        azure_endpoint="https://test.openai.azure.com/",
        api_version="2024-02-01",
        azure_deployment="gpt-35-turbo-instruct",
    )
    client1 = _get_default_httpx_client(llm1.openai_api_base, llm1.request_timeout)
    client2 = _get_default_httpx_client(llm2.openai_api_base, llm2.request_timeout)
    assert client1 is client2
