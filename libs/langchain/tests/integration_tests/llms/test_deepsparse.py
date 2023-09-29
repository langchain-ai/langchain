"""Test DeepSparse wrapper."""
from langchain.llms import DeepSparse


def test_deepsparse_call() -> None:
    """Test valid call to DeepSparse."""
    config = {"max_generated_tokens": 5, "use_deepsparse_cache": False}

    llm = DeepSparse(
        model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",
        config=config,
    )

    output = llm("def ")
    assert isinstance(output, str)
    assert len(output) > 1
    assert output == "ids_to_names"
