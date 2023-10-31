"""Test WatsonxLLM API wrapper."""

from langchain.llms import WatsonxLLM


def test_watsonxllm_call() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        credentials={"url": "https://us-south.ml.cloud.ibm.com"},
    )
    response = watsonxllm("What color sunflower is?")
    assert isinstance(response, str)
