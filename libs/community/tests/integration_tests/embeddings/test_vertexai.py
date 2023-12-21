"""Test Vertex AI API wrapper.
In order to run this test, you need to install VertexAI SDK
pip install google-cloud-aiplatform>=1.35.0

Your end-user credentials would be used to make the calls (make sure you've run
`gcloud auth login` first).
"""
from langchain_community.embeddings import VertexAIEmbeddings


def test_embedding_documents() -> None:
    documents = ["foo bar"]
    model = VertexAIEmbeddings()
    output = model.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768
    assert model.model_name == model.client._model_id


def test_embedding_query() -> None:
    document = "foo bar"
    model = VertexAIEmbeddings()
    output = model.embed_query(document)
    assert len(output) == 768


def test_large_batches() -> None:
    documents = ["foo bar" for _ in range(0, 251)]
    model_uscentral1 = VertexAIEmbeddings(location="us-central1")
    model_asianortheast1 = VertexAIEmbeddings(location="asia-northeast1")
    model_uscentral1.embed_documents(documents)
    model_asianortheast1.embed_documents(documents)
    assert model_uscentral1.instance["batch_size"] >= 250
    assert model_asianortheast1.instance["batch_size"] < 50


def test_paginated_texts() -> None:
    documents = [
        "foo bar",
        "foo baz",
        "bar foo",
        "baz foo",
        "bar bar",
        "foo foo",
        "baz baz",
        "baz bar",
    ]
    model = VertexAIEmbeddings()
    output = model.embed_documents(documents)
    assert len(output) == 8
    assert len(output[0]) == 768
    assert model.model_name == model.client._model_id
