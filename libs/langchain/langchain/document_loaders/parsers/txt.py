"""Module for parsing text files.."""
from typing import Iterator

from langchain_core.documents import Document

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob


class TextParser(BaseBlobParser):
    """Parser for text blobs."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        yield Document(page_content=blob.as_string(), metadata={"source": blob.source})
