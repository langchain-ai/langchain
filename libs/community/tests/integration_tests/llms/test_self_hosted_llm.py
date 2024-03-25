"""Test Self-hosted LLMs."""
from typing import Any

from langchain_community.llms import SelfHostedHuggingFaceLLM

model_reqs = ["transformers", "torch", "accelerate", "huggingface-hub"]


def get_remote_instance() -> Any:
    """Get remote instance for testing."""
    import runhouse as rh

    gpu = rh.cluster(name="rh-a10x", instance_type="g5.4xlarge", use_spot=False)
    gpu.run(commands=["pip install langchain"])
    return gpu


def get_remote_env(gpu: Any) -> Any:
    import runhouse as rh

    model_env = rh.env(
        name="model_env",
        reqs=model_reqs,
        secrets=["huggingface"],  # need for downloading models from huggingface
    ).to(system=gpu)

    return model_env


def test_self_hosted_huggingface_pipeline_text_generation() -> None:
    """Test valid call to self-hosted HuggingFace text generation model."""
    gpu = get_remote_instance()
    gpu.up_if_not()
    env = get_remote_env(gpu)
    llm = SelfHostedHuggingFaceLLM(
        model_id="google/gemma-2b-it",
        hardware=gpu,
        env=env,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)


def test_self_hosted_huggingface_pipeline_text2text_generation() -> None:
    """Test valid call to self-hosted HuggingFace text2text generation model."""
    gpu = get_remote_instance()
    gpu.up_if_not()
    env = get_remote_env(gpu)
    llm = SelfHostedHuggingFaceLLM(
        model_id="google/flan-t5-small",
        task="text2text-generation",
        hardware=gpu,
        env=env,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)


def test_self_hosted_huggingface_pipeline_summarization() -> None:
    """Test valid call to self-hosted HuggingFace summarization model."""
    gpu = get_remote_instance()
    gpu.up_if_not()
    env = get_remote_env(gpu)
    llm = SelfHostedHuggingFaceLLM(
        model_id="facebook/bart-large-cnn",
        task="summarization",
        hardware=gpu,
        env=env,
    )
    output = llm("Say foo:")
    assert isinstance(output, str)
