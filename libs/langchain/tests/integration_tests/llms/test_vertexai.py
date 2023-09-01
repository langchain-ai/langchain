"""Test Vertex AI API wrapper.
In order to run this test, you need to install VertexAI SDK (that is is the private
preview)  and be whitelisted to list the models themselves:
In order to run this test, you need to install VertexAI SDK 
pip install google-cloud-aiplatform>=1.25.0

Your end-user credentials would be used to make the calls (make sure you've run 
`gcloud auth login` first).
"""
import os

from langchain.llms import VertexAI, VertexAIModelGarden
from langchain.schema import LLMResult


def test_vertex_call() -> None:
    llm = VertexAI()
    output = llm("Say foo:")
    assert isinstance(output, str)
    assert llm._llm_type == "vertexai"
    assert llm.model_name == llm.client._model_id


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
    print(output)
    assert isinstance(output, str)
    assert llm._llm_type == "vertexai_model_garden"


def test_model_garden_batch() -> None:
    """In order to run this test, you should provide an endpoint name.

    Example:
    export ENDPOINT_ID=...
    export PROJECT=...
    """
    endpoint_id = os.environ["ENDPOINT_ID"]
    project = os.environ["PROJECT"]
    llm = VertexAIModelGarden(endpoint_id=endpoint_id, project=project)
    output = llm._generate(["What is the meaning of life?", "How much is 2+2"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 2
