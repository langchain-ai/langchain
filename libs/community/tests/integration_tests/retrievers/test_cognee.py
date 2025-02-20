from typing import Type

import pytest
from langchain_core.documents import Document
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)

try:
    from langchain_cognee.retrievers import CogneeRetriever
except ImportError:
    pytestmark = pytest.mark.skip(reason="langchain_cognee not installed")
    CogneeRetriever = None


class TestCogneeRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[CogneeRetriever]:
        """Get an empty vectorstore for unit tests."""
        return CogneeRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {
            "k": 3,
            "llm_api_key": "",
            "dataset_name": "test_dataset",
        }

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a dictionary representing the "args" of an example retriever call.
        """
        return "Tell me about SpaceX"

    @pytest.fixture(autouse=True)
    def setup_dataset(self) -> None:
        """
        Retrieve real data. For faster tests, remove it and mock _search_cognee instead.
        """
        retriever = self.retriever_constructor(**self.retriever_constructor_params)
        retriever.prune()
        docs = [
            Document(page_content="Elon Musk is CEO of SpaceX."),
            Document(page_content="SpaceX is focused on rockets and space travel."),
            Document(page_content="Tesla is another company led by Musk."),
            Document(page_content="SpaceX was founded in 2002."),
        ]
        retriever.add_documents(docs)
        retriever.process_data()
