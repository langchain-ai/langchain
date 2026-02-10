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


def test_azure_stop_tokens_merge(monkeypatch: Any) -> None:
    """Test that Azure OpenAI also properly merges stop tokens.

    Azure OpenAI inherits from BaseOpenAI, so it should also benefit from
    the stop token merging fix.
    """
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    llm = AzureOpenAI(
        openai_api_key="secret-api-key",  # type: ignore[call-arg]
        azure_endpoint="endpoint",
        api_version="version",
        azure_deployment="gpt-35-turbo-instruct",
        model_kwargs={"stop": ["<|eot_id|>", "<|eom_id|>"]},
    )

    # Test that model_kwargs stop tokens are preserved
    params = llm._invocation_params.copy()
    llm.get_sub_prompts(params, ["test prompt"], stop=None)
    assert "stop" in params
    assert "<|eot_id|>" in params["stop"]
    assert "<|eom_id|>" in params["stop"]

    # Test that stop tokens are merged, not overwritten
    params = llm._invocation_params.copy()
    llm.get_sub_prompts(params, ["test prompt"], stop=["STOP"])
    assert "stop" in params
    assert "<|eot_id|>" in params["stop"], "Azure: model_kwargs stop tokens should not be overwritten"
    assert "<|eom_id|>" in params["stop"], "Azure: model_kwargs stop tokens should not be overwritten"
    assert "STOP" in params["stop"], "Azure: runtime stop tokens should be included"
    assert len(params["stop"]) == 3, "Azure: should have all 3 unique stop tokens"


def test_azure_ls_params_stop_tokens_merge(monkeypatch: Any) -> None:
    """Test that Azure's _get_ls_params properly merges stop tokens for tracing.

    This is critical for accurate LangSmith tracing when stop tokens are specified
    in model_kwargs. If this test fails, it means the Azure-specific fix is missing.
    """
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    llm = AzureOpenAI(
        openai_api_key="secret-api-key",  # type: ignore[call-arg]
        azure_endpoint="endpoint",
        api_version="version",
        azure_deployment="gpt-35-turbo-instruct",
        model_kwargs={"stop": ["<|eot_id|>", "<|eom_id|>"]},
    )

    # Test that _get_ls_params merges stop tokens from model_kwargs with runtime stop
    ls_params = llm._get_ls_params(stop=["STOP"])

    # The stop parameter should be merged
    assert "ls_stop" in ls_params
    stop_tokens = ls_params["ls_stop"]
    assert "<|eot_id|>" in stop_tokens, "Azure _get_ls_params: model_kwargs stop tokens must be included"
    assert "<|eom_id|>" in stop_tokens, "Azure _get_ls_params: model_kwargs stop tokens must be included"
    assert "STOP" in stop_tokens, "Azure _get_ls_params: runtime stop tokens must be included"
    assert len(stop_tokens) == 3, "Azure _get_ls_params: should have all 3 unique stop tokens"
