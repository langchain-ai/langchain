"""Test HuggingFace Pipeline wrapper."""

from pathlib import Path

import pytest

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.loading import load_llm
from tests.integration_tests.llms.utils import assert_llm_equality


def test_huggingface_pipeline_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = HuggingFacePipeline(model_id="gpt2", model_kwargs={"max_new_tokens": 10})
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_huggingface_pipeline_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = HuggingFacePipeline(model_id="google/flan-t5-xl")
    output = llm("The capital of New York is")
    assert output == "Albany"


def test_huggingface_pipeline_call_error() -> None:
    """Test valid call to HuggingFace that errors."""
    llm = HuggingFacePipeline(model_kwargs={"max_new_tokens": -1})
    with pytest.raises(ValueError):
        llm("Say foo:")


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an HuggingFaceHub LLM."""
    llm = HuggingFacePipeline(repo_id="gpt2", model_kwargs={"max_new_tokens": 10})
    llm.save(file_path=tmp_path / "hf.yaml")
    loaded_llm = load_llm(tmp_path / "hf.yaml")
    assert_llm_equality(llm, loaded_llm)
