"""Test Azure OpenAI Chat API wrapper."""

import os
from unittest import mock

import pytest
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict

from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models._client_utils import (
    _cached_async_httpx_client,
    _cached_sync_httpx_client,
)


def test_initialize_azure_openai() -> None:
    llm = AzureChatOpenAI(  # type: ignore[call-arg]
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
        azure_endpoint="my-base-url",
    )
    assert llm.deployment_name == "35-turbo-dev"
    assert llm.openai_api_version == "2023-05-15"
    assert llm.azure_endpoint == "my-base-url"


def test_initialize_more() -> None:
    llm = AzureChatOpenAI(  # type: ignore[call-arg]
        api_key="xyz",  # type: ignore[arg-type]
        azure_endpoint="my-base-url",
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
        temperature=0,
        model="gpt-35-turbo",
        model_version="0125",
    )
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "xyz"
    assert llm.azure_endpoint == "my-base-url"
    assert llm.deployment_name == "35-turbo-dev"
    assert llm.openai_api_version == "2023-05-15"
    assert llm.temperature == 0

    ls_params = llm._get_ls_params()
    assert ls_params["ls_provider"] == "azure"
    assert ls_params["ls_model_name"] == "gpt-35-turbo-0125"


def test_initialize_azure_openai_with_openai_api_base_set() -> None:
    with mock.patch.dict(os.environ, {"OPENAI_API_BASE": "https://api.openai.com"}):
        llm = AzureChatOpenAI(  # type: ignore[call-arg, call-arg]
            api_key="xyz",  # type: ignore[arg-type]
            azure_endpoint="my-base-url",
            azure_deployment="35-turbo-dev",
            openai_api_version="2023-05-15",
            temperature=0,
            openai_api_base=None,
        )
        assert llm.openai_api_key is not None
        assert llm.openai_api_key.get_secret_value() == "xyz"
        assert llm.azure_endpoint == "my-base-url"
        assert llm.deployment_name == "35-turbo-dev"
        assert llm.openai_api_version == "2023-05-15"
        assert llm.temperature == 0

        ls_params = llm._get_ls_params()
        assert ls_params["ls_provider"] == "azure"
        assert ls_params["ls_model_name"] == "35-turbo-dev"


def test_structured_output_old_model() -> None:
    class Output(TypedDict):
        """output."""

        foo: str

    with pytest.warns(match="Cannot use method='json_schema'"):
        llm = AzureChatOpenAI(  # type: ignore[call-arg]
            model="gpt-35-turbo",
            azure_deployment="35-turbo-dev",
            openai_api_version="2023-05-15",
            azure_endpoint="my-base-url",
        ).with_structured_output(Output)

    # assert tool calling was used instead of json_schema
    assert "tools" in llm.steps[0].kwargs  # type: ignore
    assert "response_format" not in llm.steps[0].kwargs  # type: ignore


def test_max_completion_tokens_in_payload() -> None:
    llm = AzureChatOpenAI(
        azure_deployment="o1-mini",
        api_version="2024-12-01-preview",
        azure_endpoint="my-base-url",
        model_kwargs={"max_completion_tokens": 300},
    )
    messages = [HumanMessage("Hello")]
    payload = llm._get_request_payload(messages)
    assert payload == {
        "messages": [{"content": "Hello", "role": "user"}],
        "model": None,
        "stream": False,
        "max_completion_tokens": 300,
    }


def test_http_client_reuse() -> None:
    """Test that multiple AzureChatOpenAI instances reuse the same HTTP client."""
    # Clear the cache first
    _cached_sync_httpx_client.cache_clear()
    _cached_async_httpx_client.cache_clear()

    # Track cache info before creating instances
    initial_sync_cache_info = _cached_sync_httpx_client.cache_info()
    initial_async_cache_info = _cached_async_httpx_client.cache_info()

    # Create multiple instances with the same configuration
    llm1 = AzureChatOpenAI(  # type: ignore[call-arg]  # noqa: F841
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
        azure_endpoint="https://my-base-url.openai.azure.com",
    )

    llm2 = AzureChatOpenAI(  # type: ignore[call-arg]  # noqa: F841
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
        azure_endpoint="https://my-base-url.openai.azure.com",
    )

    llm3 = AzureChatOpenAI(  # type: ignore[call-arg]  # noqa: F841
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
        azure_endpoint="https://my-base-url.openai.azure.com",
    )

    # Check cache statistics to verify reuse
    final_sync_cache_info = _cached_sync_httpx_client.cache_info()
    final_async_cache_info = _cached_async_httpx_client.cache_info()

    # For sync client: first call should be a miss, subsequent calls should be hits
    assert final_sync_cache_info.misses == initial_sync_cache_info.misses + 1
    assert final_sync_cache_info.hits == initial_sync_cache_info.hits + 2

    # For async client: first call should be a miss, subsequent calls should be hits
    assert final_async_cache_info.misses == initial_async_cache_info.misses + 1
    assert final_async_cache_info.hits == initial_async_cache_info.hits + 2
