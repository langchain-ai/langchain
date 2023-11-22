"""Code for generic / auxiliary parsers.

This module contains some logic to help assemble more sophisticated parsers.
"""
from typing import Iterator, Mapping, Optional

from langchain_core.documents import Document

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders.schema import Blob


class MimeTypeBasedParser(BaseBlobParser):
    """Parser that uses `mime`-types to parse a blob.

    This parser is useful for simple pipelines where the mime-type is sufficient
    to determine how to parse a blob.

    To use, configure handlers based on mime-types and pass them to the initializer.

    Example:

        .. code-block:: python

        from langchain.document_loaders.parsers.generic import MimeTypeBasedParser

        parser = MimeTypeBasedParser(
            handlers={
                "application/pdf": ...,
            },
            fallback_parser=...,
        )
    """

    def __init__(
        self,
        handlers: Mapping[str, BaseBlobParser],
        *,
        fallback_parser: Optional[BaseBlobParser] = None,
    ) -> None:
        """Define a parser that uses mime-types to determine how to parse a blob.

        Args:
            handlers: A mapping from mime-types to functions that take a blob, parse it
                      and return a document.
            fallback_parser: A fallback_parser parser to use if the mime-type is not
                             found in the handlers. If provided, this parser will be
                             used to parse blobs with all mime-types not found in
                             the handlers.
                             If not provided, a ValueError will be raised if the
                             mime-type is not found in the handlers.
        """
        self.handlers = handlers
        self.fallback_parser = fallback_parser

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Load documents from a blob."""
        mimetype = blob.mimetype

        if mimetype is None:
            raise ValueError(f"{blob} does not have a mimetype.")

        if mimetype in self.handlers:
            handler = self.handlers[mimetype]
            yield from handler.lazy_parse(blob)
        else:
            if self.fallback_parser is not None:
                yield from self.fallback_parser.lazy_parse(blob)
            else:
                raise ValueError(f"Unsupported mime type: {mimetype}")
