"""Test WatsonxLLM API wrapper."""

from langchain_community.llms import WatsonxLLM


def test_watsonxllm_call() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        apikey="***",
        project_id="***",
    )
    response = watsonxllm.invoke("What color sunflower is?")
    assert isinstance(response, str)
