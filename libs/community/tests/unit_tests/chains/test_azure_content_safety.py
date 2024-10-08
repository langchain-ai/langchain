"""Tests for the Azure AI Content Safety Chain."""

from typing import Any

import pytest

from langchain_community.chains.azure_content_safety_chain import (
    AzureAIContentSafetyChain,
    AzureHarmfulContentError,
)


@pytest.mark.requires("azure.ai.contentsafety")
def test_content_safety(mocker: Any) -> None:
    mocker.patch("azure.ai.contentsafety.ContentSafetyClient", autospec=True)
    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)

    key = "key"
    endpoint = "endpoint"

    chain = AzureAIContentSafetyChain(
        content_safety_key=key, content_safety_endpoint=endpoint
    )
    assert chain.content_safety_key == key
    assert chain.content_safety_endpoint == endpoint


@pytest.mark.requires("azure.ai.contentsafety")
def test_raise_error_when_harmful_content_detected(mocker: Any) -> None:
    key = "key"
    endpoint = "endpoint"

    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)
    mocker.patch("azure.ai.contentsafety.ContentSafetyClient", autospec=True)
    chain = AzureAIContentSafetyChain(
        content_safety_key=key, content_safety_endpoint=endpoint, error=True
    )

    mock_content_client = mocker.Mock()
    mock_content_client.analyze_text.return_value.categories_analysis = [
        {"Category": "Harm", "severity": 1}
    ]

    chain.client = mock_content_client

    text = "This text contains harmful content"
    with pytest.raises(AzureHarmfulContentError):
        chain._call({chain.input_key: text})


@pytest.mark.requires("azure.ai.contentsafety")
def test_no_harmful_content_detected(mocker: Any) -> None:
    key = "key"
    endpoint = "endpoint"

    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)
    mocker.patch("azure.ai.contentsafety.ContentSafetyClient", autospec=True)
    chain = AzureAIContentSafetyChain(
        content_safety_key=key, content_safety_endpoint=endpoint, error=True
    )

    mock_content_client = mocker.Mock()
    mock_content_client.analyze_text.return_value.categories_analysis = [
        {"Category": "Harm", "severity": 0}
    ]

    chain.client = mock_content_client

    text = "This text contains no harmful content"
    output = chain._call({chain.input_key: text})

    assert output[chain.output_key] == text
