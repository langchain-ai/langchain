"""Test HuggingFace Pipeline wrapper."""

import os

import pytest

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


@pytest.mark.skipif(not os.getenv("RUN_HPU_TEST"), reason="RUN_HPU_TEST is not set")
def test_huggingface_pipeline_text_generation_on_hpu() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 10},
        model_kwargs={"device": "hpu"},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


@pytest.mark.skipif(not os.getenv("RUN_HPU_TEST"), reason="RUN_HPU_TEST is not set")
def test_huggingface_pipeline_text2text_generation_on_hpu() -> None:
    """Test valid call to HuggingFace text2text generation model."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small",
        task="text2text-generation",
        model_kwargs={"device": "hpu"},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


@pytest.mark.skipif(not os.getenv("RUN_HPU_TEST"), reason="RUN_HPU_TEST is not set")
def test_huggingface_pipeline_invalid_hpu_and_openvino_backend() -> None:
    """Test invalid backend."""
    try:
        HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-small",
            task="text2text-generation",
            model_kwargs={"device": "hpu", "backend": "openvino"},
        )
    except ValueError as e:
        assert """Cannot specify `model_kwargs{'device': 'hpu'}`
            and `backend=openvino` at the same time.""" in str(e)


@pytest.mark.skipif(not os.getenv("RUN_HPU_TEST"), reason="RUN_HPU_TEST is not set")
def test_huggingface_pipeline_summarization_on_hpu() -> None:
    """Test valid call to HuggingFace summarization model."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="facebook/bart-large-cnn",
        task="summarization",
        model_kwargs={"device": "hpu"},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
