"""Test Self-hosted LLMs."""
import pickle
from typing import Any, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.llms import SelfHostedHuggingFaceLLM, SelfHostedPipeline

model_reqs = ["pip:./", "transformers", "torch"]


def get_remote_instance() -> Any:
    """Get remote instance for testing."""
    import runhouse as rh

    return rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)


def test_self_hosted_huggingface_pipeline_text_generation() -> None:
    """Test valid call to self-hosted HuggingFace text generation model."""
    gpu = get_remote_instance()
    llm = SelfHostedHuggingFaceLLM(
        model_id="gpt2",
        task="text-generation",
        model_kwargs={"n_positions": 1024},
        hardware=gpu,
        model_reqs=model_reqs,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)


def test_self_hosted_huggingface_pipeline_text2text_generation() -> None:
    """Test valid call to self-hosted HuggingFace text2text generation model."""
    gpu = get_remote_instance()
    llm = SelfHostedHuggingFaceLLM(
        model_id="google/flan-t5-small",
        task="text2text-generation",
        hardware=gpu,
        model_reqs=model_reqs,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)


def test_self_hosted_huggingface_pipeline_summarization() -> None:
    """Test valid call to self-hosted HuggingFace summarization model."""
    gpu = get_remote_instance()
    llm = SelfHostedHuggingFaceLLM(
        model_id="facebook/bart-large-cnn",
        task="summarization",
        hardware=gpu,
        model_reqs=model_reqs,
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def load_pipeline() -> Any:
    """Load pipeline for testing."""
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    return pipe


def inference_fn(pipeline: Any, prompt: str, stop: Optional[List[str]] = None) -> str:
    """Inference function for testing."""
    return pipeline(prompt)[0]["generated_text"]


def test_init_with_local_pipeline() -> None:
    """Test initialization with a self-hosted HF pipeline."""
    gpu = get_remote_instance()
    pipeline = load_pipeline()
    llm = SelfHostedPipeline.from_pipeline(
        pipeline=pipeline,
        hardware=gpu,
        model_reqs=model_reqs,
        inference_fn=inference_fn,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)


def test_init_with_pipeline_path() -> None:
    """Test initialization with a self-hosted HF pipeline."""
    gpu = get_remote_instance()
    pipeline = load_pipeline()
    import runhouse as rh

    rh.blob(pickle.dumps(pipeline), path="models/pipeline.pkl").save().to(
        gpu, path="models"
    )
    llm = SelfHostedPipeline.from_pipeline(
        pipeline="models/pipeline.pkl",
        hardware=gpu,
        model_reqs=model_reqs,
        inference_fn=inference_fn,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)


def test_init_with_pipeline_fn() -> None:
    """Test initialization with a self-hosted HF pipeline."""
    gpu = get_remote_instance()
    llm = SelfHostedPipeline(
        model_load_fn=load_pipeline,
        hardware=gpu,
        model_reqs=model_reqs,
        inference_fn=inference_fn,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)
