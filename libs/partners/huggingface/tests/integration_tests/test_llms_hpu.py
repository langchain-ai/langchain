from typing import Generator

from langchain_huggingface.llms import HuggingFacePipeline


def test_huggingface_pipeline_streaming_on_hpu() -> None:
    """Test streaming tokens from huggingface_pipeline on HPU."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 10},
        model_kwargs={"device": "hpu"},
    )
    generator = llm.stream("Q: How do you say 'hello' in German? A:'", stop=["."])
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert isinstance(chunk, str)
        stream_results_string = chunk
    assert len(stream_results_string.strip()) > 1
