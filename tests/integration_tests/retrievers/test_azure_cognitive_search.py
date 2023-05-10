"""Test Azure Cognitive Search wrapper."""

import os
from typing import List

from langchain.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever
from langchain.schema import Document


def test_azure_cognitive_search_call() -> None:
    """Test valid call to Azure Cognitive Search."""

    assert os.environ.get("AZURE_COGNITIVE_SEARCH_SERVICE_NAME") is not None
    assert os.environ.get("AZURE_COGNITIVE_SEARCH_INDEX_NAME") is not None
    assert os.environ.get("AZURE_COGNITIVE_SEARCH_API_KEY") is not None

    retriever = AzureCognitiveSearchRetriever(
        service_name=os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"],
        index_name=os.environ["AZURE_COGNITIVE_SEARCH_INDEX_NAME"],
        azure_cognitive_search_api_key=os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"],
    )

    documents = retriever.get_relevant_documents("what is langchain")

    for doc in documents:
        assert isinstance(doc, Document)
