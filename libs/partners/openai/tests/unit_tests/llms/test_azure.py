from typing import Any

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
