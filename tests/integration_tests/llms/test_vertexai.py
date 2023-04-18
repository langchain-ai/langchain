"""Test VertexAI API wrapper.

In order to run this test, you need to install VertexAI SDK (that is is the private preview) and be whitelisted to list the models themselves:

gsutil cp gs://vertex_sdk_llm_private_releases/SDK/google_cloud_aiplatform-1.25.dev20230413+language.models-py2.py3-none-any.whl .
pip install invoke
pip install google_cloud_aiplatform-1.25.dev20230413+language.models-py2.py3-none-any.whl "shapely<2.0.0"

Your end-user credentials would be used to make the calls (make sure you've run `gcloud auth login` first).
"""
from langchain.llms import VertexAI


def test_vertex_call() -> None:
    llm = VertexAI()
    output = llm("Say foo:")
    assert isinstance(output, str)
    assert llm._llm_type == "text-bison-001"
