import pytest
from pytest_mock import MockerFixture

from langchain_community.retrievers.needle import NeedleRetriever


# Mock class to simulate search results from Needle API
class MockSearchResult:
    def __init__(self, content):
        self.content = content


# Mock class to simulate NeedleClient and its collections behavior
class MockNeedleClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.collections = self.MockCollections()

    class MockCollections:
        def search(self, collection_id: str, text: str):
            return [
                MockSearchResult(content="Result for query: " + text),
                MockSearchResult(content="Another result for query: " + text),
            ]


def test_needle_retriever_initialization() -> None:
    """Test that the NeedleRetriever is initialized correctly."""
    retriever = NeedleRetriever(
        needle_api_key="mock_api_key",
        collection_id="mock_collection_id",
    )

    assert retriever.needle_api_key == "mock_api_key"
    assert retriever.collection_id == "mock_collection_id"


@pytest.mark.usefixtures("socket_enabled")
def test_get_relevant_documents(mocker: MockerFixture) -> None:
    """Test that the retriever correctly fetches documents."""
    # Patch NeedleClient with the mock
    mocker.patch(
        "langchain_community.retrievers.needle.NeedleClient", new=MockNeedleClient
    )

    # Initialize the retriever with mocked API key and collection ID
    retriever = NeedleRetriever(
        needle_api_key="mock_api_key", collection_id="mock_collection_id"
    )

    # Perform the search
    query = "What is RAG?"
    retrieved_documents = retriever._get_relevant_documents(query, run_manager=None)

    # Validate the results
    assert len(retrieved_documents) == 2
    assert retrieved_documents[0].page_content == "Result for query: What is RAG?"
    assert (
        retrieved_documents[1].page_content == "Another result for query: What is RAG?"
    )
