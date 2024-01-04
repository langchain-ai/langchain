"""Test Azure Cognitive Search wrapper."""
from langchain_core.documents import Document

from langchain_community.retrievers.azure_cognitive_search import (
    AzureCognitiveSearchRetriever,
)


def test_azure_cognitive_search_get_relevant_documents() -> None:
    """Test valid call to Azure Cognitive Search."""
    retriever = AzureCognitiveSearchRetriever()
    documents = retriever.get_relevant_documents("what is langchain")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content

    retriever = AzureCognitiveSearchRetriever(top_k=1)
    documents = retriever.get_relevant_documents("what is langchain")
    assert len(documents) <= 1


async def test_azure_cognitive_search_aget_relevant_documents() -> None:
    """Test valid async call to Azure Cognitive Search."""
    retriever = AzureCognitiveSearchRetriever()
    documents = await retriever.aget_relevant_documents("what is langchain")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
