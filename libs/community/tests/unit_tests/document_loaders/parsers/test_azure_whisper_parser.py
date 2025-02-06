"""Tests for the Azure OpenAI Whisper parser."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.documents.base import Blob

from langchain_community.document_loaders.parsers.audio import AzureOpenAIWhisperParser

_THIS_DIR = Path(__file__).parents[3]

_EXAMPLES_DIR = _THIS_DIR / "examples"
AUDIO_M4A = _EXAMPLES_DIR / "hello_world.m4a"


@pytest.mark.requires("openai")
@patch("openai.AzureOpenAI")
def test_azure_openai_whisper(mock_client: Mock) -> None:
    endpoint = "endpoint"
    key = "key"
    version = "115"
    name = "model"

    parser = AzureOpenAIWhisperParser(
        api_key=key, azure_endpoint=endpoint, api_version=version, deployment_name=name
    )
    mock_client.assert_called_once_with(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=version,
        max_retries=3,
        azure_ad_token_provider=None,
    )
    assert parser._client == mock_client()


@pytest.mark.requires("openai")
def test_is_openai_v1_lazy_parse(mocker: Any) -> None:
    endpoint = "endpoint"
    key = "key"
    version = "115"
    name = "model"

    mock_blob = mocker.Mock(spec=Blob)
    mock_blob.path = AUDIO_M4A
    mock_blob.source = "test_source"

    mock_openai_client = mocker.Mock()

    mock_openai_client.audio.transcriptions.create.return_value = mocker.Mock()
    mock_openai_client.audio.transcriptions.create.return_value.text = (
        "Transcribed text"
    )

    mocker.patch("langchain_community.utils.openai.is_openai_v1", return_value=True)

    parser = AzureOpenAIWhisperParser(
        api_key=key, azure_endpoint=endpoint, api_version=version, deployment_name=name
    )

    parser._client = mock_openai_client

    result = list(parser.lazy_parse(mock_blob))

    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].page_content == "Transcribed text"
    assert result[0].metadata["source"] == "test_source"


@pytest.mark.requires("openai")
def test_is_not_openai_v1_lazy_parse(mocker: Any) -> None:
    endpoint = "endpoint"
    key = "key"
    version = "115"
    name = "model"

    mock_blob = mocker.Mock(spec=Blob)
    mock_blob.path = AUDIO_M4A
    mock_blob.source = "test_source"

    mock_openai_client = mocker.Mock()

    mock_openai_client.audio.transcriptions.create.return_value = mocker.Mock()
    mock_openai_client.audio.transcriptions.create.return_value.text = (
        "Transcribed text"
    )

    mocker.patch("langchain_community.utils.openai.is_openai_v1", return_value=False)

    parser = AzureOpenAIWhisperParser(
        api_key=key, azure_endpoint=endpoint, api_version=version, deployment_name=name
    )
    parser._client = mock_openai_client

    result = list(parser.lazy_parse(mock_blob))

    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].page_content == "Transcribed text"
    assert result[0].metadata["source"] == "test_source"
