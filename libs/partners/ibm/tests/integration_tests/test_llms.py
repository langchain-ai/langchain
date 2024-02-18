"""Test WatsonxLLM API wrapper.

You'll need to set WATSONX_APIKEY and WATSONX_PROJECT_ID environment variables.
"""

import os

from langchain_core.outputs import LLMResult

from langchain_ibm import WatsonxLLM

PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")


def test_watsonxllm_invoke() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=PROJECT_ID,
    )
    response = watsonxllm.invoke("What color sunflower is?")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_generate() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=PROJECT_ID,
    )
    response = watsonxllm.generate(["What color sunflower is?"])
    response_text = response.generations[0][0].text
    assert isinstance(response, LLMResult)
    assert len(response_text) > 0


def test_watsonxllm_generate_with_multiple_prompts() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=PROJECT_ID,
    )
    response = watsonxllm.generate(
        ["What color sunflower is?", "What color turtle is?"]
    )
    response_text = response.generations[0][0].text
    assert isinstance(response, LLMResult)
    assert len(response_text) > 0


def test_watsonxllm_generate_stream() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=PROJECT_ID,
    )
    response = watsonxllm.generate(["What color sunflower is?"], stream=True)
    response_text = response.generations[0][0].text
    assert isinstance(response, LLMResult)
    assert len(response_text) > 0


def test_watsonxllm_stream() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=PROJECT_ID,
    )
    response = watsonxllm.invoke("What color sunflower is?")

    stream_response = watsonxllm.stream("What color sunflower is?")

    linked_text_stream = ""
    for chunk in stream_response:
        assert isinstance(
            chunk, str
        ), f"chunk expect type '{str}', actual '{type(chunk)}'"
        linked_text_stream += chunk

    assert (
        response == linked_text_stream
    ), "Linked text stream are not the same as generated text"
