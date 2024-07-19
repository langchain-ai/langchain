"""Test in memory indexer"""

from typing import AsyncGenerator, Generator

import pytest
from langchain_standard_tests.integration_tests.indexer import (
    AsyncDocumentIndexerTestSuite,
    DocumentIndexerTestSuite,
)

from langchain_core.indexing import AsyncDocumentIndexer, DocumentIndexer
from langchain_core.indexing.in_memory import (
    AsyncInMemoryDocumentIndexer,
    InMemoryDocumentIndexer,
)


class TestDocumentIndexerTestSuite(DocumentIndexerTestSuite):
    @pytest.fixture()
    def indexer(self) -> Generator[DocumentIndexer, None, None]:
        yield InMemoryDocumentIndexer()


class TestAsyncDocumentIndexerTestSuite(AsyncDocumentIndexerTestSuite):
    # Something funky is going on with mypy and async pytest fixture
    @pytest.fixture()
    async def indexer(self) -> AsyncGenerator[AsyncDocumentIndexer, None]:  # type: ignore
        yield AsyncInMemoryDocumentIndexer()
