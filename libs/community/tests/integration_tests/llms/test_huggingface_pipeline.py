"""Test HuggingFace Pipeline wrapper."""

from pathlib import Path

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.loading import load_llm
from tests.integration_tests.llms.utils import assert_llm_equality


def test_huggingface_pipeline_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2", task="text-generation", pipeline_kwargs={"max_new_tokens": 10}
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_huggingface_pipeline_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text generation model."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small", task="text2text-generation"
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_huggingface_pipeline_device_map() -> None:
    """Test pipelines specifying the device map parameter."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        device_map="auto",
        pipeline_kwargs={"max_new_tokens": 10},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def text_huggingface_pipeline_summarization() -> None:
    """Test valid call to HuggingFace summarization model."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="facebook/bart-large-cnn", task="summarization"
    )
    output = llm.invoke("Say foo:")
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_huggingface_pipeline_runtime_kwargs() -> None:
    """Test pipelines specifying the device map parameter."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
    )
    prompt = "Say foo:"
    output = llm.invoke(prompt, pipeline_kwargs={"max_new_tokens": 2})
    assert len(output) < 10


ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}


def test_huggingface_pipeline_text_generation_ov() -> None:
    """Test valid call to HuggingFace text generation model with openvino."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        backend="openvino",
        model_kwargs={"device": "CPU", "ov_config": ov_config},
        pipeline_kwargs={"max_new_tokens": 64},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_huggingface_pipeline_text2text_generation_ov() -> None:
    """Test valid call to HuggingFace text2text generation model with openvino."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small",
        task="text2text-generation",
        backend="openvino",
        model_kwargs={"device": "CPU", "ov_config": ov_config},
        pipeline_kwargs={"max_new_tokens": 64},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def text_huggingface_pipeline_summarization_ov() -> None:
    """Test valid call to HuggingFace summarization model with openvino."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="facebook/bart-large-cnn",
        task="summarization",
        backend="openvino",
        model_kwargs={"device": "CPU", "ov_config": ov_config},
        pipeline_kwargs={"max_new_tokens": 64},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
