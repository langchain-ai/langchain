"""Tests for the Google Cloud DocAI parser."""
from unittest.mock import ANY, patch

import pytest

from langchain.document_loaders.parsers import DocAIParser


@pytest.mark.requires("google.cloud", "google.cloud.documentai")
def test_docai_parser_valid_processor_name() -> None:
    processor_name = "projects/123456/locations/us-central1/processors/ab123dfg"
    with patch("google.cloud.documentai.DocumentProcessorServiceClient") as test_client:
        parser = DocAIParser(processor_name=processor_name, location="us")
        test_client.assert_called_once_with(client_options=ANY, client_info=ANY)
        assert parser._processor_name == processor_name


@pytest.mark.requires("google.cloud", "google.cloud.documentai")
@pytest.mark.parametrize(
    "processor_name",
    ["projects/123456/locations/us-central1/processors/ab123dfg:publish", "ab123dfg"],
)
def test_docai_parser_invalid_processor_name(processor_name: str) -> None:
    with patch("google.cloud.documentai.DocumentProcessorServiceClient"):
        with pytest.raises(ValueError):
            _ = DocAIParser(processor_name=processor_name, location="us")
