from typing import Any

from langchain_openai import AzureOpenAI

AZURE_AD_TOKEN = "token"  # noqa: S105


def test_azure_ad_token_takes_precedence_over_api_key() -> None:
    llm = AzureOpenAI(
        openai_api_key="api_key",
        azure_ad_token=AZURE_AD_TOKEN,
        azure_endpoint="https://endpoint.com",
        api_version="2023-05-15",
    )

    assert llm.client._client._azure_ad_token == AZURE_AD_TOKEN


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
