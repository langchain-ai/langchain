import pytest
from langchain_standard_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

from langchain_chroma.vectorstores import Chroma


class TestChromaReadWriteTestSuite(ReadWriteTestSuite):
    @pytest.fixture
    def vectorstore(self) -> Chroma:
        store = Chroma(embedding_function=self.get_embeddings())
        store.reset_collection()
        return store


class TestChromaAsyncReadWriteTestSuite(AsyncReadWriteTestSuite):
    @pytest.fixture()
    async def vectorstore(self) -> Chroma:
        store = Chroma(embedding_function=self.get_embeddings())
        store.reset_collection()
        return store
