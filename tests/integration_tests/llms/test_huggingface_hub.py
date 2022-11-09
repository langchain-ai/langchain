"""Test HuggingFace API wrapper."""

import pytest

from langchain.llms.huggingface_hub import HuggingFaceHub


def test_huggingface_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"max_new_tokens": 10})
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_huggingface_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = HuggingFaceHub(repo_id="google/flan-t5-xl")
    output = llm("The capital of New York is")
    assert output == "Albany"


def test_huggingface_call_error() -> None:
    """Test valid call to HuggingFace that errors."""
    llm = HuggingFaceHub(model_kwargs={"max_new_tokens": -1})
    with pytest.raises(ValueError):
        llm("Say foo:")
