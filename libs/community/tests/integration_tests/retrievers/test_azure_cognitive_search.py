"""Test Azure Cognitive Search wrapper."""
from langchain_core.documents import Document

from langchain_community.retrievers.azure_cognitive_search import (
    AzureCognitiveSearchRetriever,
)


def test_azure_cognitive_search_get_relevant_documents() -> None:
    """Test valid call to Azure Cognitive Search.

    In order to run this test, you should provide a service name, azure search api key
    and an index_name as arguments for the AzureCognitiveSearchRetriever in both tests.
    """
    retriever = AzureCognitiveSearchRetriever()

    documents = retriever.get_relevant_documents("what is langchain?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content

    retriever = AzureCognitiveSearchRetriever()
    documents = retriever.get_relevant_documents("what is langchain?")
    assert len(documents) <= 1


async def test_azure_cognitive_search_aget_relevant_documents() -> None:
    """Test valid async call to Azure Cognitive Search.

    In order to run this test, you should provide a service name, azure search api key
    and an index_name as arguments for the AzureCognitiveSearchRetriever.
    """
    retriever = AzureCognitiveSearchRetriever()
    documents = await retriever.aget_relevant_documents("what is langchain?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
