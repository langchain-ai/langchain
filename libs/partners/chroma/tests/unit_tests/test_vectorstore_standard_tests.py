from typing import AsyncGenerator, Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_standard_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

from langchain_chroma.vectorstores import Chroma


class TestChromaReadWriteTestSuite(ReadWriteTestSuite):
    @pytest.fixture
    def vectorstore(self) -> Generator[VectorStore, None, None]:
        store = Chroma(embedding_function=self.get_embeddings())
        store.reset_collection()
        yield store


class TestChromaAsyncReadWriteTestSuite(AsyncReadWriteTestSuite):
    @pytest.fixture()
    async def vectorstore(self) -> AsyncGenerator[VectorStore, None]:
        store = Chroma(embedding_function=self.get_embeddings())
        store.reset_collection()
        yield store
