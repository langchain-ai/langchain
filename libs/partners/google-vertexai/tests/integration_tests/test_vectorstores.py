"""Test Vertex AI API wrapper.

Your end-user credentials would be used to make the calls (make sure you've run 
`gcloud auth login` first).

Additionally in order to run the test you must have set the following environment 
variables:

- PROJECT_ID: Id of the Google Cloud Project
- REGION: Region of the Bucket, Index and Endpoint
- GCS_BUCKET_NAME: Name of a Google Cloud Storage Bucket
- INDEX_ID: Id of the Vector Search index.
- ENDPOINT_ID: Id of the Vector Search endpoint.

If required to run slow tests, enviroment variable 'RUN_SLOW_TESTS' must be set

"""

import os
from typing import Dict

import pytest

from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_vertexai.vectorstores import (
    GCSVertexAIVectorSearchDocumentStorage,
    VertexAIVectorSearch,
)


@pytest.fixture
def vector_store_kwargs() -> Dict[str, str]:
    kwargs = dict(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ["REGION"],
        gcs_bucket_name=os.environ["GCS_BUCKET_NAME"],
        index_id=os.environ["INDEX_ID"],
        endpoint_id=os.environ["ENDPOINT_ID"],
        embedding=VertexAIEmbeddings(),
    )

    return kwargs


@pytest.fixture
def vector_store(vector_store_kwargs) -> VertexAIVectorSearch:
    return VertexAIVectorSearch.from_components(**vector_store_kwargs)


def test_constructor(vector_store_kwargs):
    vector_store = VertexAIVectorSearch.from_components(**vector_store_kwargs)

    assert isinstance(vector_store, VertexAIVectorSearch)


def test_similarity_search(vector_store: VertexAIVectorSearch):
    query = "What are your favourite animals?"
    docs = vector_store.similarity_search_with_score(query, k=1)
    assert len(docs) == 1


def test_add_texts(vector_store: VertexAIVectorSearch):

    if os.environ.get("RUN_SLOW_TESTS") is None:
        return 
    
    vector_store.add_texts(
        texts=[
            "Lions are my favourite animals",
            "There are two apples on the table",
            "Today is raining a lot in Madrid",
        ]
    )


def test_gcs_document_storage():

    from google.cloud import storage

    client = storage.Client(project=os.environ["PROJECT_ID"])
    bucket = client.bucket(bucket_name=os.environ["GCS_BUCKET_NAME"])

    prefix = "test-document-storage"

    storage = GCSVertexAIVectorSearchDocumentStorage(bucket=bucket, prefix=prefix)
    
    id_ = "my-id"
    text = "My text"
    storage.store_by_id(id_, text)

    assert storage.get_by_id(id_) == text

    assert storage.get_by_id("non-existent") is None
