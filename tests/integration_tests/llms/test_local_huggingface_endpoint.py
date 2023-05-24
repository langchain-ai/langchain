"""Test HuggingFace API wrapper."""

import unittest
from pathlib import Path

import pytest

from langchain.llms.loading import load_llm
from langchain.llms.local_huggingface_endpoint import LocalHuggingFaceEndpoint
from tests.integration_tests.llms.utils import assert_llm_equality


@unittest.skip("This test requires an inference endpoint.")
def test_huggingface_endpoint_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = LocalHuggingFaceEndpoint(
        endpoint_url="", task="text-generation", model_kwargs={"max_new_tokens": 10}
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


@unittest.skip("This test requires an inference endpoint.")
def test_huggingface_endpoint_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = LocalHuggingFaceEndpoint(endpoint_url="", task="text2text-generation")
    output = llm("The capital of New York is")
    assert isinstance(output, str)
