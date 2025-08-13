"""Test Azure OpenAI Chat API wrapper."""

import os
from unittest import mock

import httpx
import pytest
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict

from langchain_openai import AzureChatOpenAI


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


def test_azure_client_caching() -> None:
    """Test that the Azure OpenAI client is cached."""
    llm1 = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo",
        api_version="2023-05-15",
        azure_endpoint="https://test.openai.azure.com/",
    )
    llm2 = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo",
        api_version="2023-05-15",
        azure_endpoint="https://test.openai.azure.com/",
    )
    assert llm1.root_client._client is llm2.root_client._client

    # Different endpoint should create a different client
    llm3 = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo",
        api_version="2023-05-15",
        azure_endpoint="https://different.openai.azure.com/",
    )
    assert llm1.root_client._client is not llm3.root_client._client

    # Same endpoint with timeout=None should reuse the client
    llm4 = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo",
        api_version="2023-05-15",
        azure_endpoint="https://test.openai.azure.com/",
        timeout=None,
    )
    assert llm1.root_client._client is llm4.root_client._client

    # Different timeout should create a different client
    llm5 = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo",
        api_version="2023-05-15",
        azure_endpoint="https://test.openai.azure.com/",
        timeout=3,
    )
    assert llm1.root_client._client is not llm5.root_client._client

    # httpx.Timeout object should create a different client
    llm6 = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo",
        api_version="2023-05-15",
        azure_endpoint="https://test.openai.azure.com/",
        timeout=httpx.Timeout(timeout=60.0, connect=5.0),
    )
    assert llm1.root_client._client is not llm6.root_client._client

    # Tuple timeout should create a different client
    llm7 = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo",
        api_version="2023-05-15",
        azure_endpoint="https://test.openai.azure.com/",
        timeout=(5, 1),
    )
    assert llm1.root_client._client is not llm7.root_client._client


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


