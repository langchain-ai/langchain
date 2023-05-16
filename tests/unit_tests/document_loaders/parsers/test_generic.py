"""Module to test generic parsers."""

from typing import Iterator

import pytest

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain.schema import Document


class TestMimeBasedParser:
    """Test mime based parser."""

    def test_without_fallback_parser(self) -> None:
        class FirstCharParser(BaseBlobParser):
            def lazy_parse(self, blob: Blob) -> Iterator[Document]:
                """Extract the first character of a blob."""
                yield Document(page_content=blob.as_string()[0])

        class SecondCharParser(BaseBlobParser):
            def lazy_parse(self, blob: Blob) -> Iterator[Document]:
                """Extract the second character of a blob."""
                yield Document(page_content=blob.as_string()[1])

        parser = MimeTypeBasedParser(
            handlers={
                "text/plain": FirstCharParser(),
                "text/html": SecondCharParser(),
            },
        )

        blob = Blob(data=b"Hello World", mimetype="text/plain")
        docs = parser.parse(blob)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content == "H"

        # Check text/html handler.
        blob = Blob(data=b"Hello World", mimetype="text/html")
        docs = parser.parse(blob)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content == "e"

        blob = Blob(data=b"Hello World", mimetype="text/csv")

        with pytest.raises(ValueError, match="Unsupported mime type"):
            # Check that the fallback parser is used when the mimetype is not found.
            parser.parse(blob)

    def test_with_fallback_parser(self) -> None:
        class FirstCharParser(BaseBlobParser):
            def lazy_parse(self, blob: Blob) -> Iterator[Document]:
                """Extract the first character of a blob."""
                yield Document(page_content=blob.as_string()[0])

        class SecondCharParser(BaseBlobParser):
            def lazy_parse(self, blob: Blob) -> Iterator[Document]:
                """Extract the second character of a blob."""
                yield Document(page_content=blob.as_string()[1])

        class ThirdCharParser(BaseBlobParser):
            def lazy_parse(self, blob: Blob) -> Iterator[Document]:
                """Extract the third character of a blob."""
                yield Document(page_content=blob.as_string()[2])

        parser = MimeTypeBasedParser(
            handlers={
                "text/plain": FirstCharParser(),
                "text/html": SecondCharParser(),
            },
            fallback_parser=ThirdCharParser(),
        )

        blob = Blob(data=b"Hello World", mimetype="text/plain")
        docs = parser.parse(blob)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content == "H"

        # Check text/html handler.
        blob = Blob(data=b"Hello World", mimetype="text/html")
        docs = parser.parse(blob)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content == "e"

        # Check that the fallback parser is used when the mimetype is not found.
        blob = Blob(data=b"Hello World", mimetype="text/csv")
        docs = parser.parse(blob)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content == "l"
