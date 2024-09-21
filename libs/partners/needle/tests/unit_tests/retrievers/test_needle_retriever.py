import pytest
from pytest_mock import MockerFixture
from langchain_core.documents import Document
from langchain_needle import NeedleRetriever

# Mock class to simulate search results
class MockSearchResult:
    def __init__(self, content):
        self.content = content

# Mock class to simulate NeedleClient
class MockNeedleClient:
    class MockCollections:
        def search(self, collection_id: str, text: str):
            return [
                MockSearchResult(content="Result for query: " + text),
                MockSearchResult(content="Another result for query: " + text)
            ]

    def __init__(self):
        self.collections = self.MockCollections()

# Test the NeedleRetriever initialization
def test_needle_retriever_initialization() -> None:
    retriever = NeedleRetriever(
        needle_api_key="mock_api_key",
        collection_id="mock_collection_id"
    )

    assert retriever.needle_api_key == "mock_api_key"
    assert retriever.collection_id == "mock_collection_id"

# Need to pass real API key and collection ID to test this function, otherwise fails
@pytest.mark.usefixtures("socket_enabled")
def test_get_relevant_documents(mocker: MockerFixture) -> None:
    # Mock the NeedleClient
    mocker.patch("needle.v1.NeedleClient", new=MockNeedleClient)

    retriever = NeedleRetriever(
        needle_api_key="YOUR_API_KEY",
        collection_id="YOUR_COLLECTION_ID"
    )

    query = "What is RAG?"
    retrieved_documents = retriever._get_relevant_documents(query, run_manager=None)

    assert len(retrieved_documents) == 5
