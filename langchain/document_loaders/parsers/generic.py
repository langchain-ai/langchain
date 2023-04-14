"""Implementations of generic use parsers."""
from typing import Mapping, Callable, Generator

from langchain.document_loaders.base import Blob, BaseBlobParser
from langchain.schema import Document


class MimeTypeBasedParser(BaseBlobParser):
    """A parser that uses mime-types to determine strategy to use to parse a blob.

    This parser is useful for simple pipelines where the mime-type is sufficient to determine
    how to parse a blob.

    To use, configure handlers based on mime-types and pass them to the initializer.

    Example:

        >>> from langchain.document_loaders.parsers.generic import MimeTypeBasedParser
        >>> from langchain.document_loaders.parsers.image import ImageParser
        >>> from langchain.document_loaders.parsers.pdf import PDFParser

        >>> parser = MimeTypeBasedParser({
        ...     "image/png": ImageParser(),
        ...     "application/pdf": PDFParser(),
        ... })
        >>> parser.parse(Blob(data=b"Hello world", mimetype="text/plain"))

        [Document(page_content='Hello world', metadata={})]
    """

    def __init__(self, handlers: Mapping[str, Callable[[Blob], Document]]) -> None:
        """A parser based on mime-types.

        Args:
            handlers: A mapping from mime-types to functions that take a blob, parse it and
                      return a document.
        """
        self.handlers = handlers

    def parse(self, blob: Blob) -> Generator[Document, None, None]:
        """Load documents from a file."""

        mimetype = blob.mimetype

        if mimetype is None:
            err_msg = "Mime type must be specified provided."
            if blob.path_like is not None:
                err_msg += f" Location: {blob.path_like}"
            raise ValueError(err_msg)

        if mimetype in self.handlers:
            handler = self.handlers[mimetype]
            document = handler(blob)
            yield document
        else:
            raise ValueError(f"Unsupported mime type: {mimetype}")
