from typing import Any

import pytest
from pytest_mock import MockerFixture


# Mock class to simulate search results from Needle API
class MockSearchResult:
    def __init__(self, content: str) -> None:
        self.content = content


# Mock class to simulate NeedleClient and its collections behavior
class MockNeedleClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.collections = self.MockCollections()

    class MockCollections:
        def search(self, collection_id: str, text: str) -> list[MockSearchResult]:
            return [
                MockSearchResult(content=f"Result for query: {text}"),
                MockSearchResult(content=f"Another result for query: {text}"),
            ]


@pytest.mark.requires("needle")
def test_needle_retriever_initialization() -> None:
    """
    Test that the NeedleRetriever is initialized correctly.
    """
    from langchain_community.retrievers.needle import NeedleRetriever  # noqa: I001

    retriever = NeedleRetriever(
        needle_api_key="mock_api_key",
        collection_id="mock_collection_id",
    )

    assert retriever.needle_api_key == "mock_api_key"
    assert retriever.collection_id == "mock_collection_id"


@pytest.mark.requires("needle")
def test_get_relevant_documents(mocker: MockerFixture) -> None:
    """
    Test that the retriever correctly fetches documents.
    """
    from langchain_community.retrievers.needle import NeedleRetriever  # noqa: I001

    # Patch the actual NeedleClient import path used in the NeedleRetriever
    mocker.patch("needle.v1.NeedleClient", new=MockNeedleClient)

    # Initialize the retriever with mocked API key and collection ID
    retriever = NeedleRetriever(
        needle_api_key="mock_api_key",
        collection_id="mock_collection_id",
    )

    mock_run_manager: Any = None

    # Perform the search
    query = "What is RAG?"
    retrieved_documents = retriever._get_relevant_documents(
        query, run_manager=mock_run_manager
    )

    # Validate the results
    assert len(retrieved_documents) == 2
    assert retrieved_documents[0].page_content == "Result for query: What is RAG?"
    assert (
        retrieved_documents[1].page_content == "Another result for query: What is RAG?"
    )
