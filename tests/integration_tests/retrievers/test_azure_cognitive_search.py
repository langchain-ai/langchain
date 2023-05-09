"""Test Azure Cognitive Search wrapper."""

import os
from typing import List

from langchain.schema import Document
from langchain.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever


def test_azure_cognitive_search_call() -> None:
    """Test valid call to Azure Cognitive Search."""

    assert os.environ.get("TEST_AZURE_COGNITIVE_SEARCH_SERVICE") is not None
    assert os.environ.get("TEST_AZURE_COGNITIVE_SEARCH_INDEX") is not None
    assert os.environ.get("AZURE_COGNITIVE_SEARCH_API_KEY") is not None

    retriever = AzureCognitiveSearchRetriever(
        service_name=os.environ["TEST_AZURE_COGNITIVE_SEARCH_SERVICE"],
        index_name=os.environ["TEST_AZURE_COGNITIVE_SEARCH_INDEX"],
        azure_cognitive_search_api_key=os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"]
    )

    documents = retriever.get_relevant_documents("what is langchain")

    for doc in documents:
        assert isinstance(doc, Document)
