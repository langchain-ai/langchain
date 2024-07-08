import pytest
from langchain_core.vectorstores import VectorStore

from langchain_standard_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

# We'll need to move this dependency to core
pytest.importorskip("langchain_community")

from langchain_community.vectorstores.inmemory import (  # type: ignore # noqa
    InMemoryVectorStore,
)


class TestInMemoryVectorStore(ReadWriteTestSuite):
    @pytest.fixture
    def vectorstore(self) -> VectorStore:
        embeddings = self.get_embeddings()
        return InMemoryVectorStore(embedding=embeddings)


class TestAysncInMemoryVectorStore(AsyncReadWriteTestSuite):
    @pytest.fixture
    async def vectorstore(self) -> VectorStore:
        embeddings = self.get_embeddings()
        return InMemoryVectorStore(embedding=embeddings)
