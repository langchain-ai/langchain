"""Test MLX Pipeline wrapper."""

import pytest

from langchain_community.llms.mlx_pipeline import MLXPipeline


@pytest.mark.requires("mlx_lm")
def test_mlx_pipeline_text_generation() -> None:
    """Test valid call to MLX text generation model."""
    llm = MLXPipeline.from_model_id(
        model_id="mlx-community/quantized-gemma-2b",
        pipeline_kwargs={"max_tokens": 10},
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


@pytest.mark.requires("mlx_lm")
def test_init_with_model_and_tokenizer() -> None:
    """Test initialization with a HF pipeline."""
    from mlx_lm import load

    model, tokenizer = load("mlx-community/quantized-gemma-2b")
    llm = MLXPipeline(model=model, tokenizer=tokenizer)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


@pytest.mark.requires("mlx_lm")
def test_huggingface_pipeline_runtime_kwargs() -> None:
    """Test pipelines specifying the device map parameter."""
    llm = MLXPipeline.from_model_id(
        model_id="mlx-community/quantized-gemma-2b",
    )
    prompt = "Say foo:"
    output = llm.invoke(prompt, pipeline_kwargs={"max_tokens": 2})
    assert len(output) < 10


@pytest.mark.requires("mlx_lm")
def test_mlx_pipeline_with_params() -> None:
    """Test valid call to MLX text generation model."""
    llm = MLXPipeline.from_model_id(
        model_id="mlx-community/quantized-gemma-2b",
        pipeline_kwargs={
            "max_tokens": 10,
            "temp": 0.8,
            "verbose": False,
            "repetition_penalty": 1.1,
            "repetition_context_size": 64,
            "top_p": 0.95,
        },
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
