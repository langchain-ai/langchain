import os
from collections.abc import Generator

import pytest

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


@pytest.mark.skipif(
    os.getenv("HF_TEST_DEVICE") is None,
    reason="Set HF_TEST_DEVICE=<index> on an accelerator (XPU/CUDA) box to run.",
)
def test_from_model_id_runs_on_accelerator() -> None:
    """Load a model on an accelerator and confirm placement is not CPU."""
    device = int(os.environ["HF_TEST_DEVICE"])
    model_id = os.getenv("HF_TEST_MODEL_ID", "openai-community/gpt2")

    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device=device,
        pipeline_kwargs={"max_new_tokens": 10},
    )

    result = llm.invoke("Hello")
    assert isinstance(result, str)
    assert result.strip()

    # Confirm the model is actually on the accelerator, not silently on CPU —
    # a green run with a CPU fallback would prove nothing about XPU coverage.
    model_device = llm.pipeline.model.device
    assert model_device.type != "cpu", (
        f"Expected model on accelerator, but it loaded on {model_device}. "
        "Check torch device ordering, HF_TEST_BACKEND, or device index."
    )
