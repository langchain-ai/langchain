"""Test HuggingFace Pipeline wrapper."""

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.callbacks.base import CallbackManager
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms.loading import load_llm
from tests.integration_tests.llms.utils import assert_llm_equality
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_huggingface_pipeline_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2", task="text-generation", pipeline_kwargs={"max_new_tokens": 10}
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_huggingface_pipeline_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text generation model."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small", task="text2text-generation"
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an HuggingFaceHub LLM."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2", task="text-generation", pipeline_kwargs={"max_new_tokens": 10}
    )
    llm.save(file_path=tmp_path / "hf.yaml")
    loaded_llm = load_llm(tmp_path / "hf.yaml")
    assert_llm_equality(llm, loaded_llm)


def test_init_with_pipeline() -> None:
    """Test initialization with a HF pipeline."""
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_pipeline_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 10},
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    llm("This sentence has 100 words:")
    assert callback_handler.llm_streams == 10


def test_stop() -> None:
    """Test stop logic."""
    query = "an ordered list of five items:\n1. item one\n2."
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 10, "do_sample": False},
    )
    output = llm(query, stop=["3"])
    assert output.endswith("\n")
    assert "3" not in output
