from __future__ import annotations
from typing import Iterator, Union, Optional, Sequence
from pathlib import Path

from langchain.document_loaders.base import BaseLoader, BaseBlobParser
from langchain.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain.document_loaders.blob_loaders import BlobLoader
from langchain.schema import Document

PathLike = Union[str, Path]


class GenericLoader(BaseLoader):
    """A generic document loader.

    Examples:

        ..



    """

    def __init__(
        self,
        blob_loader: BlobLoader,
        base_blob_parser: BaseBlobParser,
    ) -> None:
        """A generic document loader."""
        self.blob_loader = blob_loader
        self.blob_parser = base_blob_parser

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        blobs = self.blob_loader.yield_blobs()
        for blob in blobs:
            yield from self.blob_parser.lazy_parse(blob)

    @classmethod
    def from_filesystem(
        cls,
        path: PathLike,
        *,
        glob: str = "**/[!.]*",
        suffixes: Optional[Sequence[str]] = None,
        show_progress: bool = False,

    ) -> GenericLoader:
        """"""
        blob_loader = FileSystemBlobLoader(
            path, glob=glob, suffixes=suffixes, show_progress=show_progress
        )
        blob_parser = BaseBlobParser()
        return cls(blob_loader, blob_parser)
