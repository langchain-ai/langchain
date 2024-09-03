"""Test in memory indexer"""

from typing import AsyncGenerator, Generator

import pytest
from langchain_standard_tests.integration_tests.indexer import (
    AsyncDocumentIndexTestSuite,
    DocumentIndexerTestSuite,
)

from langchain_core.documents import Document
from langchain_core.indexing.base import DocumentIndex
from langchain_core.indexing.in_memory import (
    InMemoryDocumentIndex,
)


class TestDocumentIndexerTestSuite(DocumentIndexerTestSuite):
    @pytest.fixture()
    def index(self) -> Generator[DocumentIndex, None, None]:
        yield InMemoryDocumentIndex()


class TestAsyncDocumentIndexerTestSuite(AsyncDocumentIndexTestSuite):
    # Something funky is going on with mypy and async pytest fixture
    @pytest.fixture()
    async def index(self) -> AsyncGenerator[DocumentIndex, None]:  # type: ignore
        yield InMemoryDocumentIndex()


def test_sync_retriever() -> None:
    index = InMemoryDocumentIndex()
    documents = [
        Document(id="1", page_content="hello world"),
        Document(id="2", page_content="goodbye cat"),
    ]
    index.upsert(documents)
    assert index.invoke("hello") == [documents[0], documents[1]]
    assert index.invoke("cat") == [documents[1], documents[0]]


async def test_async_retriever() -> None:
    index = InMemoryDocumentIndex()
    documents = [
        Document(id="1", page_content="hello world"),
        Document(id="2", page_content="goodbye cat"),
    ]
    await index.aupsert(documents)
    assert (await index.ainvoke("hello")) == [documents[0], documents[1]]
    assert (await index.ainvoke("cat")) == [documents[1], documents[0]]
