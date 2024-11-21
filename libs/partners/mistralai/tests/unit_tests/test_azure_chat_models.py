"""Test MistralAI Chat API wrapper."""

import pytest
from typing import Tuple, Type
import os
from unittest import mock

from langchain_mistralai import AzureChatMistralAI


def test_initialize_azure_mistralai() -> None:
    llm = AzureChatMistralAI(
        azure_endpoint="my-base-url",
    )
    assert llm.azure_endpoint == "my-base-url"


def test_initialize_more() -> None:
    llm = AzureChatMistralAI(  # type: ignore[call-arg]
        api_key="xyz",
        azure_endpoint="my-base-url",
        temperature=0,
        model="mistral-small",
    )

    assert llm.mistral_api_key is not None
    assert llm.mistral_api_key.get_secret_value() == "xyz"
    assert llm.azure_endpoint == "my-base-url"
    assert llm.temperature == 0
    assert llm.model == "mistral-small"

    ls_params = llm._get_ls_params()
    assert ls_params["ls_model_name"] == "mistral-small"


def test_initialize_azure_mistral_with_mistral_endpoint_and_key_in_env() -> None:
    with mock.patch.dict(os.environ, {
        "AZURE_MISTRAL_ENDPOINT": "https://azure.mistral-endpoint.com/",
        "AZURE_MISTRAL_API_KEY": "env-secret-key",
    }):
        llm = AzureChatMistralAI()
        assert llm.azure_endpoint == "https://azure.mistral-endpoint.com/"
        assert llm.mistral_api_key.get_secret_value() == "env-secret-key"
