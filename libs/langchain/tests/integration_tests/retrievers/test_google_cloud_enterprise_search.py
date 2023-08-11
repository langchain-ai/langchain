"""Test Google Cloud Enterprise Search retriever.

You need to create a Gen App Builder search app and populate it 
with data to run the integration tests.
Follow the instructions in the example notebook:
google_cloud_enterprise_search.ipynb
to set up the app and configure authentication.

Set the following environment variables before the tests:
PROJECT_ID - set to your Google Cloud project ID
SEARCH_ENGINE_ID - the ID of the search engine to use for the test
"""

import pytest

from langchain.retrievers.google_cloud_enterprise_search import (
    GoogleCloudEnterpriseSearchRetriever,
)
from langchain.schema import Document


@pytest.mark.requires("google_api_core")
def test_google_cloud_enterprise_search_get_relevant_documents() -> None:
    """Test the get_relevant_documents() method."""
    retriever = GoogleCloudEnterpriseSearchRetriever()
    documents = retriever.get_relevant_documents("What are Alphabet's Other Bets?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
