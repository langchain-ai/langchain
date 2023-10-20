"""Test Vertex AI API wrapper.
In order to run this test, you need to install VertexAI SDK (that is is the private
preview)  and be whitelisted to list the models themselves:
In order to run this test, you need to install VertexAI SDK 
pip install google-cloud-aiplatform>=1.25.0

Your end-user credentials would be used to make the calls (make sure you've run 
`gcloud auth login` first).
"""
import os

import pytest

from langchain.llms import VertexAI, VertexAIModelGarden
from langchain.schema import LLMResult


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
@pytest.mark.asyncio
async def test_vertex_agenerate() -> None:
    llm = VertexAI(temperature=0)
    output = await llm.agenerate(["Please say foo:"])
    assert isinstance(output, LLMResult)


@pytest.mark.scheduled
def test_vertex_stream() -> None:
    llm = VertexAI(temperature=0)
    outputs = list(llm.stream("Please say foo:"))
    assert isinstance(outputs[0], str)


@pytest.mark.asyncio
async def test_vertex_consistency() -> None:
    llm = VertexAI(temperature=0)
    output = llm.generate(["Please say foo:"])
    streaming_output = llm.generate(["Please say foo:"], stream=True)
    async_output = await llm.agenerate(["Please say foo:"])
    assert output.generations[0][0].text == streaming_output.generations[0][0].text
    assert output.generations[0][0].text == async_output.generations[0][0].text


def test_model_garden() -> None:
    """In order to run this test, you should provide an endpoint name.

    Example:
    export ENDPOINT_ID=...
    export PROJECT=...
    """
    endpoint_id = os.environ["ENDPOINT_ID"]
    project = os.environ["PROJECT"]
    llm = VertexAIModelGarden(endpoint_id=endpoint_id, project=project)
    output = llm("What is the meaning of life?")
    assert isinstance(output, str)
    assert llm._llm_type == "vertexai_model_garden"


def test_model_garden_generate() -> None:
    """In order to run this test, you should provide an endpoint name.

    Example:
    export ENDPOINT_ID=...
    export PROJECT=...
    """
    endpoint_id = os.environ["ENDPOINT_ID"]
    project = os.environ["PROJECT"]
    llm = VertexAIModelGarden(endpoint_id=endpoint_id, project=project)
    output = llm.generate(["What is the meaning of life?", "How much is 2+2"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 2


@pytest.mark.asyncio
async def test_model_garden_agenerate() -> None:
    endpoint_id = os.environ["ENDPOINT_ID"]
    project = os.environ["PROJECT"]
    llm = VertexAIModelGarden(endpoint_id=endpoint_id, project=project)
    output = await llm.agenerate(["What is the meaning of life?", "How much is 2+2"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 2
