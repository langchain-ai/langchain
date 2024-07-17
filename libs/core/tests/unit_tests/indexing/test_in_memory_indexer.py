"""Test in memory indexer"""

from typing import Generator

import pytest

from langchain_core.indexing import DocumentIndexer
from langchain_core.indexing.in_memory import InMemoryIndexer
from langchain_standard_tests.integration_tests.indexer import (
    BaseDocumentIndexerTestSuite,
)


class TestDocumentIndexerTestSuite(BaseDocumentIndexerTestSuite):
    @pytest.fixture()
    def indexer(self) -> Generator[DocumentIndexer, None, None]:
        return InMemoryIndexer()
