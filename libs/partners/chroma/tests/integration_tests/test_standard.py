from typing import AsyncGenerator, Generator

import pytest
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

from langchain_chroma import Chroma


class TestSync(ReadWriteTestSuite):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        embeddings = DeterministicFakeEmbedding(size=10)
        store = Chroma(embedding_function=embeddings)
        try:
            yield store
        finally:
            store.delete_collection()
            pass


class TestAsync(AsyncReadWriteTestSuite):
    @pytest.fixture()
    async def vectorstore(self) -> AsyncGenerator[VectorStore, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        embeddings = DeterministicFakeEmbedding(size=10)
        store = Chroma(embedding_function=embeddings)
        try:
            yield store
        finally:
            store.delete_collection()
            pass
