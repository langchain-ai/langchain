"""Tests for the Google Cloud DocAI parser."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_community.document_loaders.parsers import (
    AzureAIDocumentIntelligenceParser,
)


@pytest.mark.requires("azure", "azure.ai", "azure.ai.documentintelligence")
@patch("azure.ai.documentintelligence.DocumentIntelligenceClient")
@patch("azure.core.credentials.AzureKeyCredential")
def test_doc_intelligence(mock_credential: MagicMock, mock_client: MagicMock) -> None:
    endpoint = "endpoint"
    key = "key"

    parser = AzureAIDocumentIntelligenceParser(api_endpoint=endpoint, api_key=key)
    mock_credential.assert_called_once_with(key)
    mock_client.assert_called_once_with(
        endpoint=endpoint,
        credential=mock_credential(),
        headers={
            "x-ms-useragent": "langchain-parser/1.0.0",
        },
        features=None,
    )
    assert parser.client == mock_client()
    assert parser.api_model == "prebuilt-layout"
    assert parser.mode == "markdown"


@pytest.mark.requires("azure", "azure.ai", "azure.ai.documentintelligence")
@patch("azure.ai.documentintelligence.DocumentIntelligenceClient")
@patch("azure.core.credentials.AzureKeyCredential")
def test_doc_intelligence_with_analysis_features(
    mock_credential: MagicMock, mock_client: MagicMock
) -> None:
    endpoint = "endpoint"
    key = "key"

    analysis_features = ["ocrHighResolution", "barcodes"]
    parser = AzureAIDocumentIntelligenceParser(
        api_endpoint=endpoint, api_key=key, analysis_features=analysis_features
    )
    mock_credential.assert_called_once_with(key)
    mock_client.assert_called_once_with(
        endpoint=endpoint,
        credential=mock_credential(),
        headers={
            "x-ms-useragent": "langchain-parser/1.0.0",
        },
        features=analysis_features,
    )
    assert parser.client == mock_client()
    assert parser.api_model == "prebuilt-layout"
    assert parser.mode == "markdown"

    with pytest.raises(ValueError):
        analysis_features = ["invalid"]
        parser = AzureAIDocumentIntelligenceParser(
            api_endpoint=endpoint, api_key=key, analysis_features=analysis_features
        )
