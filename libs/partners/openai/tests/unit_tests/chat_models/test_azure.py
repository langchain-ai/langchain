"""Test Azure OpenAI Chat API wrapper."""

import os
from unittest import mock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import SecretStr
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


def test_responses_api_uses_deployment_name() -> None:
    """Test that Azure deployment name is used for Responses API."""
    llm = AzureChatOpenAI(
        azure_deployment="your_deployment",
        api_version="2025-04-01-preview",
        azure_endpoint="your_endpoint",
        api_key=SecretStr("your_api_key"),
        # Force Responses API usage by including a Responses-only parameter
        use_responses_api=True,
        output_version="responses/v1",
    )
    messages = [HumanMessage("Hello")]
    payload = llm._get_request_payload(messages)

    # For Responses API, the model field should be the deployment name
    assert payload["model"] == "your_deployment"
    assert "input" in payload  # Responses API uses 'input' instead of 'messages'


def test_chat_completions_api_uses_model_name() -> None:
    """Test that regular Chat Completions API still uses model name."""
    llm = AzureChatOpenAI(
        azure_deployment="your_deployment",
        model="gpt-5",  # This is the OpenAI model name
        api_version="2025-04-01-preview",
        azure_endpoint="your_endpoint",
        api_key=SecretStr("your_api_key"),
        # No Responses-only parameters, so Chat Completions API will be used
    )
    messages = [HumanMessage("Hello")]
    payload = llm._get_request_payload(messages)

    # For Chat Completions API, the model field should still be None/model_name
    # Azure Chat Completions uses deployment in the URL, not in the model field
    assert payload["model"] == "gpt-5"
    assert "messages" in payload  # Chat Completions API uses 'messages'
    assert "input" not in payload
