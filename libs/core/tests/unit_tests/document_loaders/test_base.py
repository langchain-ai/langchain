"""Test Base Schema of documents."""

from collections.abc import Iterator
from contextlib import aclosing

import pytest
from typing_extensions import override

from langchain_core.document_loaders.base import BaseBlobParser, BaseLoader
from langchain_core.documents import Document
from langchain_core.documents.base import Blob


def test_base_blob_parser() -> None:
    """Verify that the eager method is hooked up to the lazy method by default."""

    class MyParser(BaseBlobParser):
        """A simple parser that returns a single document."""

        @override
        def lazy_parse(self, blob: Blob) -> Iterator[Document]:
            """Lazy parsing interface."""
            yield Document(
                page_content="foo",
            )

    parser = MyParser()

    assert isinstance(parser.lazy_parse(Blob(data="who?")), Iterator)

    # We're verifying that the eager method is hooked up to the lazy method by default.
    docs = parser.parse(Blob(data="who?"))
    assert len(docs) == 1
    assert docs[0].page_content == "foo"


def test_default_lazy_load() -> None:
    class FakeLoader(BaseLoader):
        @override
        def load(self) -> list[Document]:
            return [
                Document(page_content="foo"),
                Document(page_content="bar"),
            ]

    loader = FakeLoader()
    docs = list(loader.lazy_load())
    assert docs == [Document(page_content="foo"), Document(page_content="bar")]


def test_lazy_load_not_implemented() -> None:
    class FakeLoader(BaseLoader):
        pass

    loader = FakeLoader()
    with pytest.raises(NotImplementedError):
        loader.lazy_load()


async def test_default_aload() -> None:
    class FakeLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield from [
                Document(page_content="foo"),
                Document(page_content="bar"),
            ]

    loader = FakeLoader()
    docs = loader.load()
    assert docs == [Document(page_content="foo"), Document(page_content="bar")]
    assert docs == [doc async for doc in loader.alazy_load()]
    assert docs == await loader.aload()


async def test_default_alazy_load_closes_iterator_on_aclose() -> None:
    closed = False

    class ResourceLoader(BaseLoader):
        def _iter(self) -> Iterator[Document]:
            nonlocal closed
            try:
                yield Document(page_content="first")
                yield Document(page_content="second")
            finally:
                closed = True

        @override
        def lazy_load(self) -> Iterator[Document]:
            return self._iter()

    loader = ResourceLoader()
    async with aclosing(loader.alazy_load()) as docs:
        first = await anext(docs)
        assert first.page_content == "first"
    assert closed is True


async def test_default_alazy_load_handles_iterator_without_close() -> None:
    class PlainIterator:
        def __init__(self) -> None:
            self.index = 0

        def __iter__(self) -> Iterator[Document]:
            return self

        def __next__(self) -> Document:
            if self.index >= 2:
                raise StopIteration
            self.index += 1
            return Document(page_content=f"doc-{self.index}")

    class PlainLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            return PlainIterator()

    loader = PlainLoader()
    async with aclosing(loader.alazy_load()) as docs:
        first = await anext(docs)
        assert first.page_content == "doc-1"
