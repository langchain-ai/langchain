"""Test Vertex AI API wrapper.
In order to run this test, you need to install VertexAI SDK:
pip install google-cloud-aiplatform>=1.36.0

Your end-user credentials would be used to make the calls (make sure you've run 
`gcloud auth login` first).
"""
import os
from typing import Optional

import pytest
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from pytest_mock import MockerFixture

from langchain.chains.summarize import load_summarize_chain
from langchain.llms import VertexAI, VertexAIModelGarden


def test_vertex_initialization() -> None:
    llm = VertexAI()
    assert llm._llm_type == "vertexai"
    assert llm.model_name == llm.client._model_id


def test_vertex_call() -> None:
    llm = VertexAI(temperature=0)
    output = llm("Say foo:")
    assert isinstance(output, str)


@pytest.mark.scheduled
def test_vertex_generate() -> None:
    llm = VertexAI(temperature=0.3, n=2, model_name="text-bison@001")
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == 2


@pytest.mark.scheduled
def test_vertex_generate_code() -> None:
    llm = VertexAI(temperature=0.3, n=2, model_name="code-bison@001")
    output = llm.generate(["generate a python method that says foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == 2


@pytest.mark.scheduled
async def test_vertex_agenerate() -> None:
    llm = VertexAI(temperature=0)
    output = await llm.agenerate(["Please say foo:"])
    assert isinstance(output, LLMResult)


@pytest.mark.scheduled
def test_vertex_stream() -> None:
    llm = VertexAI(temperature=0)
    outputs = list(llm.stream("Please say foo:"))
    assert isinstance(outputs[0], str)


async def test_vertex_consistency() -> None:
    llm = VertexAI(temperature=0)
    output = llm.generate(["Please say foo:"])
    streaming_output = llm.generate(["Please say foo:"], stream=True)
    async_output = await llm.agenerate(["Please say foo:"])
    assert output.generations[0][0].text == streaming_output.generations[0][0].text
    assert output.generations[0][0].text == async_output.generations[0][0].text


@pytest.mark.parametrize(
    "endpoint_os_variable_name,result_arg",
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
def test_model_garden(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    """In order to run this test, you should provide endpoint names.

    Example:
    export FALCON_ENDPOINT_ID=...
    export LLAMA_ENDPOINT_ID=...
    export PROJECT=...
    """
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT"]
    location = "europe-west4"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = llm("What is the meaning of life?")
    assert isinstance(output, str)
    assert llm._llm_type == "vertexai_model_garden"


@pytest.mark.parametrize(
    "endpoint_os_variable_name,result_arg",
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
def test_model_garden_generate(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    """In order to run this test, you should provide endpoint names.

    Example:
    export FALCON_ENDPOINT_ID=...
    export LLAMA_ENDPOINT_ID=...
    export PROJECT=...
    """
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT"]
    location = "europe-west4"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = llm.generate(["What is the meaning of life?", "How much is 2+2"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint_os_variable_name,result_arg",
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
async def test_model_garden_agenerate(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT"]
    location = "europe-west4"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = await llm.agenerate(["What is the meaning of life?", "How much is 2+2"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 2


def test_vertex_call_count_tokens() -> None:
    llm = VertexAI()
    output = llm.get_num_tokens("How are you?")
    assert output == 4


@pytest.mark.requires("google.cloud.aiplatform")
def test_get_num_tokens_be_called_when_using_mapreduce_chain(
    mocker: MockerFixture,
) -> None:
    from vertexai.language_models._language_models import CountTokensResponse

    m1 = mocker.patch(
        "vertexai.preview.language_models._PreviewTextGenerationModel.count_tokens",
        return_value=CountTokensResponse(
            total_tokens=2,
            total_billable_characters=2,
            _count_tokens_response={"total_tokens": 2, "total_billable_characters": 2},
        ),
    )
    llm = VertexAI()
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        return_intermediate_steps=False,
    )
    doc = Document(page_content="Hi")
    output = chain({"input_documents": [doc]})
    assert isinstance(output["output_text"], str)
    m1.assert_called_once()
    assert llm._llm_type == "vertexai"
    assert llm.model_name == llm.client._model_id
