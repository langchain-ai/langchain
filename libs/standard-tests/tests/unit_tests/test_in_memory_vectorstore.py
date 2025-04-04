import pytest
from langchain_core.vectorstores import (
    InMemoryVectorStore,
    VectorStore,
)

from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests


class TestInMemoryVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(self) -> VectorStore:
        embeddings = self.get_embeddings()
        return InMemoryVectorStore(embedding=embeddings)
