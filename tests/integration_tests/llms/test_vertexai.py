"""Test Vertex AI API wrapper.
In order to run this test, you need to install VertexAI SDK (that is is the private
preview)  and be whitelisted to list the models themselves:
In order to run this test, you need to install VertexAI SDK 
pip install google-cloud-aiplatform>=1.25.0

Your end-user credentials would be used to make the calls (make sure you've run 
`gcloud auth login` first).
"""
from langchain.llms import VertexAI


def test_vertex_call() -> None:
    llm = VertexAI()
    output = llm("Say foo:")
    assert isinstance(output, str)
    assert llm._llm_type == "vertexai"
    assert llm.model_name == llm.client._model_id
