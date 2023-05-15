"""Test Azure Cognitive Search wrapper."""
import pytest

from langchain.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever
from langchain.schema import Document


def test_azure_cognitive_search_get_relevant_documents() -> None:
    """Test valid call to Azure Cognitive Search."""
    retriever = AzureCognitiveSearchRetriever()
    documents = retriever.get_relevant_documents("what is langchain")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content


@pytest.mark.asyncio
async def test_azure_cognitive_search_aget_relevant_documents() -> None:
    """Test valid async call to Azure Cognitive Search."""
    retriever = AzureCognitiveSearchRetriever()
    documents = await retriever.aget_relevant_documents("what is langchain")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
