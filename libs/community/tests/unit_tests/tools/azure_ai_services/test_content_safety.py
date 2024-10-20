"""Tests for the Azure AI Content Safety Text Tool."""

from typing import Any

import pytest

from langchain_community.tools.azure_ai_services.content_safety import (
    AzureContentSafetyTextTool,
)


@pytest.mark.requires("azure.ai.contentsafety")
def test_content_safety(mocker: Any) -> None:
    mocker.patch("azure.ai.contentsafety.ContentSafetyClient", autospec=True)
    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)

    key = "key"
    endpoint = "endpoint"

    tool = AzureContentSafetyTextTool(
        content_safety_key=key, content_safety_endpoint=endpoint
    )
    assert tool.content_safety_key == key
    assert tool.content_safety_endpoint == endpoint


@pytest.mark.requires("azure.ai.contentsafety")
def test_harmful_content_detected(mocker: Any) -> None:
    key = "key"
    endpoint = "endpoint"

    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)
    mocker.patch("azure.ai.contentsafety.ContentSafetyClient", autospec=True)
    tool = AzureContentSafetyTextTool(
        content_safety_key=key, content_safety_endpoint=endpoint
    )

    mock_content_client = mocker.Mock()
    mock_content_client.analyze_text.return_value.categories_analysis = [
        {"category": "Harm", "severity": 1}
    ]

    tool.content_safety_client = mock_content_client

    input = "This text contains harmful content"
    output = "Harm: 1\n"

    result = tool._run(input)
    assert result == output


@pytest.mark.requires("azure.ai.contentsafety")
def test_no_harmful_content_detected(mocker: Any) -> None:
    key = "key"
    endpoint = "endpoint"

    tool = AzureContentSafetyTextTool(
        content_safety_key=key, content_safety_endpoint=endpoint
    )

    mock_content_client = mocker.Mock()
    mock_content_client.analyze_text.return_value.categories_analysis = [
        {"category": "Harm", "severity": 0}
    ]

    tool.content_safety_client = mock_content_client

    input = "This text contains harmful content"
    output = "Harm: 0\n"

    result = tool._run(input)
    assert result == output
