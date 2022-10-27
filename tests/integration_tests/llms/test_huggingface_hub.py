"""Test HuggingFace API wrapper."""

import pytest

from langchain.llms.huggingface_hub import HuggingFaceHub


def test_huggingface_call() -> None:
    """Test valid call to HuggingFace."""
    llm = HuggingFaceHub(max_new_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_huggingface_call_error() -> None:
    """Test valid call to HuggingFace that errors."""
    llm = HuggingFaceHub(max_new_tokens=-1)
    with pytest.raises(ValueError):
        llm("Say foo:")
