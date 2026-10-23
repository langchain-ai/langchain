import os
from collections.abc import Callable, Generator

import torch

from langchain_huggingface.llms import HuggingFacePipeline


def test_huggingface_pipeline_streaming() -> None:
    """Test streaming tokens from huggingface_pipeline."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="openai-community/gpt2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 10},
    )
    generator = llm.stream("Q: How do you say 'hello' in German? A:'", stop=["."])
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert isinstance(chunk, str)
        stream_results_string = chunk
    assert len(stream_results_string.strip()) > 0


def test_from_model_id_runs_on_accelerator(
    accelerator_device: int | str,
    assert_on_accelerator: Callable[[torch.device], None],
) -> None:
    """Load a model on an accelerator and confirm placement is not CPU."""
    model_id = os.getenv("HF_TEST_MODEL_ID", "openai-community/gpt2")

    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device=accelerator_device,
        pipeline_kwargs={"max_new_tokens": 10},
    )

    result = llm.invoke("Hello")
    assert isinstance(result, str)
    assert result.strip()
    assert_on_accelerator(llm.pipeline.model.device)


def test_from_model_id_keeps_device_map_on_accelerator(
    accelerator_device: int | str,
    assert_on_accelerator: Callable[[torch.device], None],
) -> None:
    """Load a model with a single-device `device_map` and confirm it stays there."""
    model_id = os.getenv("HF_TEST_MODEL_ID", "openai-community/gpt2")

    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device_map=accelerator_device,
        pipeline_kwargs={"max_new_tokens": 10},
    )

    assert llm.invoke("Hello").strip()
    assert_on_accelerator(llm.pipeline.model.device)
