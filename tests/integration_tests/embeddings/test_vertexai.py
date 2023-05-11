"""Test Vertex AI API wrapper.
In order to run this test, you need to install VertexAI SDK (that is is the private
preview)  and be whitelisted to list the models themselves:
export SDK=google_cloud_aiplatform-1.25.dev20230413+language.models-py2.py3-none-any.whl
gsutil cp gs://vertex_sdk_llm_private_releases/SDK/${SDK SDK} .
pip install invoke
pip install ${SDK} "shapely<2.0.0"
Your end-user credentials would be used to make the calls (make sure you've run 
`gcloud auth login` first).
"""
from langchain.embeddings import VertexAIEmbeddings


def test_embedding_documents() -> None:
    documents = ["foo bar"]
    model = VertexAIEmbeddings()
    output = model.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768
    assert model._llm_type == "vertexai"
    assert model.model_name == model.client._model_id


def test_embedding_query() -> None:
    """Test llamacpp embeddings."""
    document = "foo bar"
    model = VertexAIEmbeddings()
    output = model.embed_query(document)
    assert len(output) == 768
