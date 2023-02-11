"""Test HuggingFace API wrapper."""

import unittest
from pathlib import Path

import pytest

from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.llms.loading import load_llm
from tests.integration_tests.llms.utils import assert_llm_equality


@unittest.skip(
    "This test requires an inference endpoint. Tested with Hugging Face endpoints"
)
def test_huggingface_endpoint_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = HuggingFaceEndpoint(
        endpoint_url="", task="text-generation", model_kwargs={"max_new_tokens": 10}
    )
    output = llm("Say foo:")
    print(output)
    assert isinstance(output, str)


@unittest.skip(
    "This test requires an inference endpoint. Tested with Hugging Face endpoints"
)
def test_huggingface_endpoint_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = HuggingFaceEndpoint(endpoint_url="", task="text2text-generation")
    output = llm("The capital of New York is")
    assert output == "Albany"


def test_huggingface_endpoint_call_error() -> None:
    """Test valid call to HuggingFace that errors."""
    llm = HuggingFaceEndpoint(model_kwargs={"max_new_tokens": -1})
    with pytest.raises(ValueError):
        llm("Say foo:")


def test_saving_loading_endpoint_llm(tmp_path: Path) -> None:
    """Test saving/loading an HuggingFaceHub LLM."""
    llm = HuggingFaceEndpoint(
        endpoint_url="", task="text-generation", model_kwargs={"max_new_tokens": 10}
    )
    llm.save(file_path=tmp_path / "hf.yaml")
    loaded_llm = load_llm(tmp_path / "hf.yaml")
    assert_llm_equality(llm, loaded_llm)
