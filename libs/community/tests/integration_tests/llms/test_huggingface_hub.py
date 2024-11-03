"""Test HuggingFace API wrapper."""

from pathlib import Path

import pytest

from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.loading import load_llm
from tests.integration_tests.llms.utils import assert_llm_equality


def test_huggingface_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"max_new_tokens": 10})  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_huggingface_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = HuggingFaceHub(repo_id="google/flan-t5-xl")  # type: ignore[call-arg]
    output = llm.invoke("The capital of New York is")
    assert output == "Albany"


def test_huggingface_summarization() -> None:
    """Test valid call to HuggingFace summarization model."""
    llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn")  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_huggingface_call_error() -> None:
    """Test valid call to HuggingFace that errors."""
    llm = HuggingFaceHub(model_kwargs={"max_new_tokens": -1})  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        llm.invoke("Say foo:")


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an HuggingFaceHub LLM."""
    llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"max_new_tokens": 10})  # type: ignore[call-arg]
    llm.save(file_path=tmp_path / "hf.yaml")
    loaded_llm = load_llm(tmp_path / "hf.yaml")
    assert_llm_equality(llm, loaded_llm)
