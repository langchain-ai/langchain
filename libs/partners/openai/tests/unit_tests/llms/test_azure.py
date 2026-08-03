from typing import Any

from langchain_openai import AzureOpenAI


def test_azure_client_caching() -> None:
    """Test that the underlying httpx client is cached and reused."""
    llm1 = AzureOpenAI(
        openai_api_key="secret-api-key",  # type: ignore[call-arg]
        azure_endpoint="endpoint",
        api_version="version",
        azure_deployment="gpt-35-turbo-instruct",
    )
    llm2 = AzureOpenAI(
        openai_api_key="secret-api-key",  # type: ignore[call-arg]
        azure_endpoint="endpoint",
        api_version="version",
        azure_deployment="gpt-35-turbo-instruct",
    )
    assert llm1.client._client._client is llm2.client._client._client
    assert llm1.async_client._client._client is llm2.async_client._client._client

    llm3 = AzureOpenAI(
        openai_api_key="secret-api-key",  # type: ignore[call-arg]
        azure_endpoint="other-endpoint",
        api_version="version",
        azure_deployment="gpt-35-turbo-instruct",
    )
    assert llm1.client._client._client is not llm3.client._client._client


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
