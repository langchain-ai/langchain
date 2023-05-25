"""Test HuggingFace API wrapper."""

import unittest

import pytest

from langchain.llms import LocalHuggingFaceEndpoint


def test_local_huggingface_endpoint_task_error() -> None:
    """Test task error raised."""
    with pytest.raises(ValueError):
        _ = LocalHuggingFaceEndpoint(task="invalid-task")


def test_local_huggingface_endpoint_url_error() -> None:
    """Test url error raised."""
    with pytest.raises(ValueError):
        _ = LocalHuggingFaceEndpoint(config_endpoint_url="", task="text-generation")


@unittest.skip("This test requires an inference endpoint.")
def test_local_huggingface_endpoint_text_generation() -> None:
    """Test valid call to HuggingFace text generation model. Tested locally."""
    llm = LocalHuggingFaceEndpoint(
        completion_endpoint_url="",
        config_endpoint_url="",
        task="text-generation",
        model_kwargs={"max_new_tokens": 10},
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


@unittest.skip("This test requires an inference endpoint.")
def test_huggingface_endpoint_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text model. Tested locally."""
    llm = LocalHuggingFaceEndpoint(
        completion_endpoint_url="",
        config_endpoint_url="",
        task="text2text-generation",
        model_kwargs={"max_new_tokens": 10},
    )
    output = llm("The capital of New York is")
    assert isinstance(output, str)
