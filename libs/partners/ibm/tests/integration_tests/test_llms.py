"""Test WatsonxLLM API wrapper.

You'll need to set WATSONX_APIKEY and WATSONX_PROJECT_ID environment variables.
"""

import os

from ibm_watsonx_ai.foundation_models import Model, ModelInference  # type: ignore
from ibm_watsonx_ai.foundation_models.utils.enums import (  # type: ignore
    DecodingMethods,
    ModelTypes,
)
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames  # type: ignore
from langchain_core.outputs import LLMResult

from langchain_ibm import WatsonxLLM

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")


def test_watsonxllm_invoke() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_with_params() -> None:
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 5,
    }

    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
        params=parameters,
    )
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_generate() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.generate(["What color sunflower is?"])
    print(f"\nResponse: {response}")
    response_text = response.generations[0][0].text
    print(f"Response text: {response_text}")
    assert isinstance(response, LLMResult)
    assert len(response_text) > 0


def test_watsonxllm_generate_with_multiple_prompts() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.generate(
        ["What color sunflower is?", "What color turtle is?"]
    )
    print(f"\nResponse: {response}")
    response_text = response.generations[0][0].text
    print(f"Response text: {response_text}")
    assert isinstance(response, LLMResult)
    assert len(response_text) > 0


def test_watsonxllm_generate_stream() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.generate(["What color sunflower is?"], stream=True)
    print(f"\nResponse: {response}")
    response_text = response.generations[0][0].text
    print(f"Response text: {response_text}")
    assert isinstance(response, LLMResult)
    assert len(response_text) > 0


def test_watsonxllm_stream() -> None:
    watsonxllm = WatsonxLLM(
        model_id="google/flan-ul2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=WX_PROJECT_ID,
    )
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")

    stream_response = watsonxllm.stream("What color sunflower is?")

    linked_text_stream = ""
    for chunk in stream_response:
        assert isinstance(
            chunk, str
        ), f"chunk expect type '{str}', actual '{type(chunk)}'"
        linked_text_stream += chunk
    print(f"Linked text stream: {linked_text_stream}")
    assert (
        response == linked_text_stream
    ), "Linked text stream are not the same as generated text"


def test_watsonxllm_invoke_from_wx_model() -> None:
    model = Model(
        model_id="google/flan-ul2",
        credentials={
            "apikey": WX_APIKEY,
            "url": "https://us-south.ml.cloud.ibm.com",
        },
        project_id=WX_PROJECT_ID,
    )
    watsonxllm = WatsonxLLM(watsonx_model=model)
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_from_wx_model_inference() -> None:
    model = ModelInference(
        model_id="google/flan-ul2",
        credentials={
            "apikey": WX_APIKEY,
            "url": "https://us-south.ml.cloud.ibm.com",
        },
        project_id=WX_PROJECT_ID,
    )
    watsonxllm = WatsonxLLM(watsonx_model=model)
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_from_wx_model_inference_with_params() -> None:
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 10,
        GenTextParamsMetaNames.TEMPERATURE: 0.5,
        GenTextParamsMetaNames.TOP_K: 50,
        GenTextParamsMetaNames.TOP_P: 1,
    }
    model = ModelInference(
        model_id="google/flan-ul2",
        credentials={
            "apikey": WX_APIKEY,
            "url": "https://us-south.ml.cloud.ibm.com",
        },
        project_id=WX_PROJECT_ID,
        params=parameters,
    )
    watsonxllm = WatsonxLLM(watsonx_model=model)
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0


def test_watsonxllm_invoke_from_wx_model_inference_with_params_as_enum() -> None:
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: DecodingMethods.GREEDY,
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 10,
        GenTextParamsMetaNames.TEMPERATURE: 0.5,
        GenTextParamsMetaNames.TOP_K: 50,
        GenTextParamsMetaNames.TOP_P: 1,
    }
    model = ModelInference(
        model_id=ModelTypes.FLAN_UL2,
        credentials={
            "apikey": WX_APIKEY,
            "url": "https://us-south.ml.cloud.ibm.com",
        },
        project_id=WX_PROJECT_ID,
        params=parameters,
    )
    watsonxllm = WatsonxLLM(watsonx_model=model)
    response = watsonxllm.invoke("What color sunflower is?")
    print(f"\nResponse: {response}")
    assert isinstance(response, str)
    assert len(response) > 0
