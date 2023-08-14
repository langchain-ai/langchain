"""Test DeepSparse wrapper."""
import pytest

from langchain.llms import DeepSparse


def test_deepsparse_call() -> None:
    """Test valid call to DeepSparse."""
    config = {"max_generated_tokens": 5}

    llm = DeepSparse(
        model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",
        config=config,
    )

    output = llm("Say foo:")
    assert isinstance(output, str)
    assert len(output) > 1
    assert 0 < len(output) <= config["max_generated_tokens"]

test_deepsparse_call()
