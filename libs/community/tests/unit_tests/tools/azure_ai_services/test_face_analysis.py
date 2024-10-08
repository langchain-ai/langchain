"""Tests for the Azure AI Face Analysis Tool."""

from pathlib import Path
from typing import Any

import pytest

from langchain_community.tools.azure_ai_services.face_analysis import (
    AzureAIFaceAnalysisTool,
)

_THIS_DIR = Path(__file__).parents[3]

_EXAMPLES_DIR = _THIS_DIR / "examples"
FACE = _EXAMPLES_DIR / "face.jpg"
NO_FACE = _EXAMPLES_DIR / "no_face.jpg"


@pytest.mark.requires("azure.ai.vision.face")
def test_face_analysis(mocker: Any) -> None:
    mocker.patch("azure.ai.vision.face.FaceClient", autospec=True)
    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)

    key = "key"
    endpoint = "endpoint"

    tool = AzureAIFaceAnalysisTool(
        azure_ai_face_key=key, azure_ai_face_endpoint=endpoint
    )
    assert tool.azure_ai_face_key == key
    assert tool.azure_ai_face_endpoint == endpoint


@pytest.mark.requires("azure.ai.vision.face")
def test_faces_detected(mocker: Any) -> None:
    key = "key"
    endpoint = "endpoint"

    mocker.patch("azure.ai.vision.face.FaceClient", autospec=True)
    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)
    tool = AzureAIFaceAnalysisTool(
        azure_ai_face_key=key, azure_ai_face_endpoint=endpoint
    )

    mock_content_client = mocker.Mock()
    mock_content_client.detect.return_value = [
        {"faceId": "69d017e8-464e-42c1-adfb-5b7fec0526ea"}
    ]

    tool.face_client = mock_content_client

    input: str = str(FACE)
    output = "FACE: 1\nfaceId: 69d017e8-464e-42c1-adfb-5b7fec0526ea\n"

    result = tool._run(input)
    assert result == output


@pytest.mark.requires("azure.ai.vision.face")
def test_no_faces_detected(mocker: Any) -> None:
    key = "key"
    endpoint = "endpoint"

    mocker.patch("azure.ai.vision.face.FaceClient", autospec=True)
    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)
    tool = AzureAIFaceAnalysisTool(
        azure_ai_face_key=key, azure_ai_face_endpoint=endpoint
    )

    mock_content_client = mocker.Mock()
    mock_content_client.detect.return_value = []

    tool.face_client = mock_content_client

    input: str = str(NO_FACE)
    output = "No faces found"

    result = tool._run(input)
    assert result == output
