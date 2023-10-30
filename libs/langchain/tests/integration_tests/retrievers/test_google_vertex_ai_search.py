"""Test Google Vertex AI Search retriever.

You need to create a Vertex AI Search app and populate it 
with data to run the integration tests.
Follow the instructions in the example notebook:
google_vertex_ai_search.ipynb
to set up the app and configure authentication.

Set the following environment variables before the tests:
export PROJECT_ID=... - set to your Google Cloud project ID
export DATA_STORE_ID=... - the ID of the search engine to use for the test
"""

import os

import pytest

from langchain.retrievers.google_vertex_ai_search import (
    GoogleCloudEnterpriseSearchRetriever,
    GoogleVertexAIMultiTurnSearchRetriever,
    GoogleVertexAISearchRetriever,
)
from langchain.schema import Document


@pytest.mark.requires("google_api_core")
def test_google_vertex_ai_search_get_relevant_documents() -> None:
    """Test the get_relevant_documents() method."""
    retriever = GoogleVertexAISearchRetriever()
    documents = retriever.get_relevant_documents("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]


@pytest.mark.requires("google_api_core")
def test_google_vertex_ai_multiturnsearch_get_relevant_documents() -> None:
    """Test the get_relevant_documents() method."""
    retriever = GoogleVertexAIMultiTurnSearchRetriever()
    documents = retriever.get_relevant_documents("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]


@pytest.mark.requires("google_api_core")
def test_google_vertex_ai_search_enterprise_search_deprecation() -> None:
    """Test the deprecation of GoogleCloudEnterpriseSearchRetriever."""
    with pytest.warns(
        DeprecationWarning,
        match="GoogleCloudEnterpriseSearchRetriever is deprecated, use GoogleVertexAISearchRetriever",  # noqa: E501
    ):
        retriever = GoogleCloudEnterpriseSearchRetriever()

    os.environ["SEARCH_ENGINE_ID"] = os.getenv("DATA_STORE_ID", "data_store_id")
    with pytest.warns(
        DeprecationWarning,
        match="The `search_engine_id` parameter is deprecated. Use `data_store_id` instead.",  # noqa: E501
    ):
        retriever = GoogleCloudEnterpriseSearchRetriever()

    # Check that mapped methods still work.
    documents = retriever.get_relevant_documents("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]
