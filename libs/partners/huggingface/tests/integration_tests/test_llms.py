from collections.abc import Generator

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
