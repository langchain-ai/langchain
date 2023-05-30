"""Test OpenLLM wrapper."""
from langchain.llms.openllm import OpenLLM


def test_openllm_runner() -> None:
    llm = OpenLLM.for_model(model_name="flan-t5", model_id="google/flan-t5-small")
    output = llm("Say foo:")
    assert isinstance(output, str)
