"""Test HuggingFace Pipeline wrapper."""
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.llms.self_hosted_hf_pipeline import SelfHostedHuggingFacePipeline


def get_remote_instance() -> Any:
    import runhouse as rh

    return rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)


def test_selfhosted_huggingface_pipeline_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    gpu = get_remote_instance()
    gpu.restart_grpc_server(resync_rh=True, restart_ray=False)
    llm = SelfHostedHuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        model_kwargs={"max_new_tokens": 10},
        hardware=gpu,
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_selfhosted_huggingface_pipeline_text2text_generation() -> None:
    """Test valid call to HuggingFace text2text generation model."""
    gpu = get_remote_instance()
    llm = SelfHostedHuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small", task="text2text-generation", hardware=gpu
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def load_pipeline() -> Any:
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    return pipe


def test_init_with_pipeline() -> None:
    """Test initialization with a HF pipeline."""
    gpu = get_remote_instance()
    llm = SelfHostedHuggingFacePipeline(
        model_load_fn=load_pipeline, model_id="gpt_2", hardware=gpu
    )
    output = llm("Say foo:")
    assert isinstance(output, str)
