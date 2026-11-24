"""Test Base Schema of documents."""

from collections.abc import Iterator
from contextlib import aclosing
from pathlib import Path

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


async def test_alazy_load_closes_iterator_when_closed_early(tmp_path: Path) -> None:
    """Verify `alazy_load` closes the sync iterator if it is not exhausted."""

    class LineLoader(BaseLoader):
        def __init__(self, path: Path) -> None:
            self.source = path.open(encoding="utf-8")
            self.iterator = self.read_lines()

        def read_lines(self) -> Iterator[Document]:
            with self.source:
                for line in self.source:
                    yield Document(page_content=line.rstrip("\n"))

        @override
        def lazy_load(self) -> Iterator[Document]:
            return self.iterator

    path = tmp_path / "docs.txt"
    path.write_text("first\nsecond\n", encoding="utf-8")
    loader = LineLoader(path)
    async with aclosing(loader.alazy_load()) as docs:
        assert (await anext(docs)).page_content == "first"

    assert loader.source.closed


async def test_alazy_load_iterator_without_close() -> None:
    """Verify `alazy_load` works with iterators that have no `close` method."""

    class ListLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            return iter([Document(page_content="foo"), Document(page_content="bar")])

    loader = ListLoader()
    async with aclosing(loader.alazy_load()) as docs:
        assert (await anext(docs)).page_content == "foo"

    assert [doc.page_content async for doc in loader.alazy_load()] == ["foo", "bar"]
