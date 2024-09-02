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


def test_azure_ai_search_invoke_managed_identity() -> None:
    """Test valid call to Azure AI Search using managed identity.

    In order to run this test, you should provide
    `service_name`,index_name`, and `azure_ad_token_provider`,
    as arguments for the AzureAISearchRetriever in both tests.
    api_version, aiosession and topk_k are optional parameters.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    bearer_token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://search.azure.com/.default"
    )

    retriever = AzureAISearchRetriever(azure_ad_token_provider=bearer_token_provider)

    documents = retriever.invoke("what is langchain?")
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content

    retriever = AzureAISearchRetriever(
        azure_ad_token_provider=bearer_token_provider, top_k=1
    )

    documents = retriever.invoke("what is langchain?")
    assert len(documents) <= 1
