"""Test HuggingFace Endpoints."""

from pathlib import Path

import pytest

from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.loading import load_llm
from tests.integration_tests.llms.utils import assert_llm_equality


def test_huggingface_endpoint_call_error() -> None:
    """Test valid call to HuggingFace that errors."""
    llm = HuggingFaceEndpoint(endpoint_url="", model_kwargs={"max_new_tokens": -1})
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


def test_huggingface_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = HuggingFaceEndpoint(repo_id="gpt2", model_kwargs={"max_new_tokens": 10})
    output = llm("Say foo:")
    print(output)  # noqa: T201
    assert isinstance(output, str)


def test_huggingface_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = HuggingFaceEndpoint(repo_id="google/flan-t5-xl")
    output = llm("The capital of New York is")
    assert output == "Albany"


def test_huggingface_summarization() -> None:
    """Test valid call to HuggingFace summarization model."""
    llm = HuggingFaceEndpoint(repo_id="facebook/bart-large-cnn")
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_huggingface_call_error() -> None:
    """Test valid call to HuggingFace that errors."""
    llm = HuggingFaceEndpoint(repo_id="gpt2", model_kwargs={"max_new_tokens": -1})
    with pytest.raises(ValueError):
        llm("Say foo:")


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an HuggingFaceEndpoint LLM."""
    llm = HuggingFaceEndpoint(repo_id="gpt2", model_kwargs={"max_new_tokens": 10})
    llm.save(file_path=tmp_path / "hf.yaml")
    loaded_llm = load_llm(tmp_path / "hf.yaml")
    assert_llm_equality(llm, loaded_llm)


def test_invocation_params_stop_sequences() -> None:
    llm = HuggingFaceEndpoint()
    assert llm._default_params["stop_sequences"] == []

    runtime_stop = None
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == []
    assert llm._default_params["stop_sequences"] == []

    runtime_stop = ["stop"]
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == ["stop"]
    assert llm._default_params["stop_sequences"] == []

    llm = HuggingFaceEndpoint(stop_sequences=["."])
    runtime_stop = ["stop"]
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == [".", "stop"]
    assert llm._default_params["stop_sequences"] == ["."]
