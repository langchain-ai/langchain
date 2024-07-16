"""Test Azure AI Search wrapper."""

from langchain_core.documents import Document

from langchain_community.retrievers.azure_ai_search import (
    AzureAISearchRetriever,
    AzureCognitiveSearchRetriever,
)


def test_azure_ai_search_invoke() -> None:
    """Test valid call to Azure AI Search.

    In order to run this test, you should provide
    a `service_name`, azure search `api_key` and an `index_name`
    as arguments for the AzureAISearchRetriever in both tests.
    api_version, aiosession and topk_k are optional parameters.
    """
    retriever = AzureAISearchRetriever()

    documents = retriever.invoke("what is langchain?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content

    retriever = AzureAISearchRetriever(top_k=1)
    documents = retriever.invoke("what is langchain?")
    assert len(documents) <= 1


async def test_azure_ai_search_ainvoke() -> None:
    """Test valid async call to Azure AI Search.

    In order to run this test, you should provide
    a `service_name`, azure search `api_key` and an `index_name`
    as arguments for the AzureAISearchRetriever.
    """
    retriever = AzureAISearchRetriever()
    documents = await retriever.ainvoke("what is langchain?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content


def test_azure_cognitive_search_invoke() -> None:
    """Test valid call to Azure Cognitive Search.

    This is to test backwards compatibility of the retriever
    """
    retriever = AzureCognitiveSearchRetriever()

    documents = retriever.invoke("what is langchain?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content

    retriever = AzureCognitiveSearchRetriever(top_k=1)
    documents = retriever.invoke("what is langchain?")
    assert len(documents) <= 1


async def test_azure_cognitive_search_ainvoke() -> None:
    """Test valid async call to Azure Cognitive Search.

    This is to test backwards compatibility of the retriever
    """
    retriever = AzureCognitiveSearchRetriever()
    documents = await retriever.ainvoke("what is langchain?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
