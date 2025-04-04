"""Tests for the Azure AI Services Image Analysis Tool."""

from pathlib import Path
from typing import Any

import pytest

from langchain_community.tools.azure_ai_services.image_analysis import (
    AzureAiServicesImageAnalysisTool,
)

this_dir = Path(__file__).parents[3]

examples_dir = this_dir / "examples"
building_path = examples_dir / "building.jpg"


@pytest.mark.requires("azure.ai.vision.imageanalysis")
def test_content_safety(mocker: Any) -> None:
    mocker.patch("azure.ai.vision.imageanalysis.ImageAnalysisClient", autospec=True)
    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)

    key = "key"
    endpoint = "endpoint"

    tool = AzureAiServicesImageAnalysisTool(
        azure_ai_services_key=key, azure_ai_services_endpoint=endpoint
    )
    assert tool.azure_ai_services_key == key
    assert tool.azure_ai_services_endpoint == endpoint


@pytest.mark.requires("azure.ai.vision.imageanalysis")
def test_local_image_analysis(mocker: Any) -> None:
    key = "key"
    endpoint = "endpoint"

    mocker.patch("azure.ai.vision.imageanalysis.ImageAnalysisClient", autospec=True)
    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)
    mocker.patch(
        "langchain_community.tools.azure_ai_services.utils.detect_file_src_type",
        return_value="local",
    )

    tool = AzureAiServicesImageAnalysisTool(
        azure_ai_services_key=key,
        azure_ai_services_endpoint=endpoint,
        visual_features=["CAPTION"],
    )

    mock_content_client = mocker.Mock()
    mock_content_client.analyze.return_value = mocker.Mock()
    mock_content_client.analyze.return_value.caption.text = "A building corner."

    mock_content_client.analyze.return_value.objects = None
    mock_content_client.analyze.return_value.tags = None
    mock_content_client.analyze.return_value.read = None
    mock_content_client.analyze.return_value.dense_captions = None
    mock_content_client.analyze.return_value.smart_crops = None
    mock_content_client.analyze.return_value.people = None

    tool.image_analysis_client = mock_content_client

    input = str(building_path)
    output = "Caption: A building corner."

    result = tool._run(input)
    assert result == output


@pytest.mark.requires("azure.ai.vision.imageanalysis")
def test_local_image_different_features(mocker: Any) -> None:
    key = "key"
    endpoint = "endpoint"

    mocker.patch("azure.ai.vision.imageanalysis.ImageAnalysisClient", autospec=True)
    mocker.patch("azure.core.credentials.AzureKeyCredential", autospec=True)
    mocker.patch(
        "langchain_community.tools.azure_ai_services.utils.detect_file_src_type",
        return_value="local",
    )

    tool = AzureAiServicesImageAnalysisTool(
        azure_ai_services_key=key,
        azure_ai_services_endpoint=endpoint,
        visual_features=["PEOPLE", "CAPTION", "SMARTCROPS"],
    )

    mock_content_client = mocker.Mock()
    mock_content_client.analyze.return_value = mocker.Mock()
    mock_content_client.analyze.return_value.caption.text = "A building corner."

    mock_content_client.analyze.return_value.objects = None
    mock_content_client.analyze.return_value.tags = None
    mock_content_client.analyze.return_value.read = None
    mock_content_client.analyze.return_value.dense_captions = None

    mock_smart_crops = mocker.MagicMock()
    mock_smart_crops.list = [
        {"aspectRatio": 1.97, "boundingBox": {"x": 43, "y": 24, "w": 853, "h": 432}}
    ]
    mock_smart_crops.__len__.return_value = 1
    mock_content_client.analyze.return_value.smart_crops = mock_smart_crops

    mock_people = mocker.MagicMock()
    mock_people.list = [
        {
            "boundingBox": {"x": 454, "y": 44, "w": 408, "h": 531},
            "confidence": 0.9601945281028748,
        },
    ]
    mock_people.__len__.return_value = 1
    mock_content_client.analyze.return_value.people = mock_people

    tool.image_analysis_client = mock_content_client

    input = str(building_path)
    output = (
        "Caption: A building corner.\n"
        "Smart Crops: {'aspectRatio': 1.97,"
        " 'boundingBox': {'x': 43, 'y': 24, 'w': 853, 'h': 432}}\n"
        "People: {'boundingBox': {'x': 454, 'y': 44, 'w': 408, 'h': 531},"
        " 'confidence': 0.9601945281028748}"
    )

    result = tool._run(input)
    assert result == output
