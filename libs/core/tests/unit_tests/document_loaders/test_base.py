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


def test_blob_as_bytes_io_with_string_data() -> None:
    """Blob.as_bytes_io() should handle string data the same way as_bytes() does."""
    blob = Blob.from_data("hello")
    with blob.as_bytes_io() as f:
        assert f.read() == b"hello"


def test_blob_as_bytes_io_with_bytes_data() -> None:
    """Blob.as_bytes_io() should handle bytes data."""
    blob = Blob.from_data(b"hello")
    with blob.as_bytes_io() as f:
        assert f.read() == b"hello"


def test_blob_as_bytes_io_with_encoding() -> None:
    """Blob.as_bytes_io() should respect the encoding parameter for string data."""
    blob = Blob(data="café", encoding="utf-8")
    with blob.as_bytes_io() as f:
        assert f.read() == "café".encode("utf-8")


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
