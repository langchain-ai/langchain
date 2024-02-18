"""Test Base Schema of documents."""
from typing import Iterator, List

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseBlobParser, BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob


def test_base_blob_parser() -> None:
    """Verify that the eager method is hooked up to the lazy method by default."""

    class MyParser(BaseBlobParser):
        """A simple parser that returns a single document."""

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


async def test_default_aload() -> None:
    class FakeLoader(BaseLoader):
        def load(self) -> List[Document]:
            return list(self.lazy_load())

        def lazy_load(self) -> Iterator[Document]:
            yield from [
                Document(page_content="foo"),
                Document(page_content="bar"),
            ]

    loader = FakeLoader()
    docs = loader.load()
    assert docs == [Document(page_content="foo"), Document(page_content="bar")]
    assert docs == [doc async for doc in loader.alazy_load()]
