import pytest
from langchain_core.vectorstores import (
    InMemoryVectorStore,
    VectorStore,
)

from langchain_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)


class TestInMemoryVectorStore(ReadWriteTestSuite):
    @pytest.fixture
    def vectorstore(self) -> VectorStore:
        embeddings = self.get_embeddings()
        return InMemoryVectorStore(embedding=embeddings)


class TestAsyncInMemoryVectorStore(AsyncReadWriteTestSuite):
    @pytest.fixture
    async def vectorstore(self) -> VectorStore:
        embeddings = self.get_embeddings()
        return InMemoryVectorStore(embedding=embeddings)
