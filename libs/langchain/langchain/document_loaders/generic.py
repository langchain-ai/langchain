from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Literal, Optional, Sequence, Union

from langchain_core.documents import Document

from langchain.document_loaders.base import BaseBlobParser, BaseLoader
from langchain.document_loaders.blob_loaders import BlobLoader, FileSystemBlobLoader
from langchain.document_loaders.parsers.registry import get_parser
from langchain.text_splitter import TextSplitter

_PathLike = Union[str, Path]

DEFAULT = Literal["default"]


class GenericLoader(BaseLoader):
    """Generic Document Loader.

    A generic document loader that allows combining an arbitrary blob loader with
    a blob parser.

    Examples:

       .. code-block:: python

        from langchain.document_loaders import GenericLoader
        from langchain.document_loaders.blob_loaders import FileSystemBlobLoader

        loader = GenericLoader.from_filesystem(
            path="path/to/directory",
            glob="**/[!.]*",
            suffixes=[".pdf"],
            show_progress=True,
        )

        docs = loader.lazy_load()
        next(docs)

        Example instantiations to change which files are loaded:

        .. code-block:: python

            # Recursively load all text files in a directory.
            loader = GenericLoader.from_filesystem("/path/to/dir", glob="**/*.txt")

            # Recursively load all non-hidden files in a directory.
            loader = GenericLoader.from_filesystem("/path/to/dir", glob="**/[!.]*")

            # Load all files in a directory without recursion.
            loader = GenericLoader.from_filesystem("/path/to/dir", glob="*")

        Example instantiations to change which parser is used:

        .. code-block:: python

            from langchain.document_loaders.parsers.pdf import PyPDFParser

            # Recursively load all text files in a directory.
            loader = GenericLoader.from_filesystem(
                "/path/to/dir",
                glob="**/*.pdf",
                parser=PyPDFParser()
            )
    """

    def __init__(
        self,
        blob_loader: BlobLoader,
        blob_parser: BaseBlobParser,
    ) -> None:
        """A generic document loader.

        Args:
            blob_loader: A blob loader which knows how to yield blobs
            blob_parser: A blob parser which knows how to parse blobs into documents
        """
        self.blob_loader = blob_loader
        self.blob_parser = blob_parser

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Load documents lazily. Use this when working at a large scale."""
        for blob in self.blob_loader.yield_blobs():
            yield from self.blob_parser.lazy_parse(blob)

    def load(self) -> List[Document]:
        """Load all documents."""
        return list(self.lazy_load())

    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        """Load all documents and split them into sentences."""
        raise NotImplementedError(
            "Loading and splitting is not yet implemented for generic loaders. "
            "When they will be implemented they will be added via the initializer. "
            "This method should not be used going forward."
        )

    @classmethod
    def from_filesystem(
        cls,
        path: _PathLike,
        *,
        glob: str = "**/[!.]*",
        exclude: Sequence[str] = (),
        suffixes: Optional[Sequence[str]] = None,
        show_progress: bool = False,
        parser: Union[DEFAULT, BaseBlobParser] = "default",
    ) -> GenericLoader:
        """Create a generic document loader using a filesystem blob loader.

        Args:
            path: The path to the directory to load documents from.
            glob: The glob pattern to use to find documents.
            suffixes: The suffixes to use to filter documents. If None, all files
                      matching the glob will be loaded.
            exclude: A list of patterns to exclude from the loader.
            show_progress: Whether to show a progress bar or not (requires tqdm).
                           Proxies to the file system loader.
            parser: A blob parser which knows how to parse blobs into documents

        Returns:
            A generic document loader.
        """
        blob_loader = FileSystemBlobLoader(
            path,
            glob=glob,
            exclude=exclude,
            suffixes=suffixes,
            show_progress=show_progress,
        )
        if isinstance(parser, str):
            blob_parser = get_parser(parser)
        else:
            blob_parser = parser
        return cls(blob_loader, blob_parser)
