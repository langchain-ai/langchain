"""Test Base Schema of documents."""

from collections.abc import Iterator

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


def test_as_bytes_io_with_bytes_data() -> None:
    """as_bytes_io works with bytes data — must not regress."""
    blob = Blob.from_data(b"hello bytes")
    with blob.as_bytes_io() as f:
        assert f.read() == b"hello bytes"


def test_as_bytes_io_with_string_data() -> None:
    """as_bytes_io must encode string data to bytes using utf-8."""
    blob = Blob.from_data("hello string")
    with blob.as_bytes_io() as f:
        assert f.read() == b"hello string"


def test_as_bytes_io_with_string_data_custom_encoding() -> None:
    """as_bytes_io must respect custom encoding when string data is used."""
    blob = Blob(data="héllo", encoding="latin-1")
    with blob.as_bytes_io() as f:
        assert f.read() == "héllo".encode("latin-1")


def test_as_bytes_io_string_consistent_with_as_bytes() -> None:
    """as_bytes_io for string data must produce the same bytes as as_bytes()."""
    text = "consistency check"
    blob = Blob.from_data(text)
    with blob.as_bytes_io() as f:
        stream_bytes = f.read()
    assert stream_bytes == blob.as_bytes()
