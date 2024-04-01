from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseBlobParser, BaseLoader
from langchain_community.document_loaders.blob_loaders import (
    BlobLoader,
    FileSystemBlobLoader,
)
from langchain_community.document_loaders.parsers.registry import get_parser

if TYPE_CHECKING:
    from langchain_text_splitters import TextSplitter

_PathLike = Union[str, Path]

DEFAULT = Literal["default"]


class GenericLoader(BaseLoader):
    """Generic Document Loader.

    A generic document loader that allows combining an arbitrary blob loader with
    a blob parser.

    Examples:

        Parse a specific PDF file:

        .. code-block:: python

            from langchain_community.document_loaders import GenericLoader
            from langchain_community.document_loaders.parsers.pdf import PyPDFParser

            # Recursively load all text files in a directory.
            loader = GenericLoader.from_filesystem(
                "my_lovely_pdf.pdf",
                parser=PyPDFParser()
            )

       .. code-block:: python

            from langchain_community.document_loaders import GenericLoader
            from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader


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

        from langchain_community.document_loaders.parsers.pdf import PyPDFParser

        # Recursively load all text files in a directory.
        loader = GenericLoader.from_filesystem(
            "/path/to/dir",
            glob="**/*.pdf",
            parser=PyPDFParser()
        )

    """  # noqa: E501

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
        parser_kwargs: Optional[dict] = None,
    ) -> GenericLoader:
        """Create a generic document loader using a filesystem blob loader.

        Args:
            path: The path to the directory to load documents from OR the path to a
                  single file to load. If this is a file, glob, exclude, suffixes
                    will be ignored.
            glob: The glob pattern to use to find documents.
            suffixes: The suffixes to use to filter documents. If None, all files
                      matching the glob will be loaded.
            exclude: A list of patterns to exclude from the loader.
            show_progress: Whether to show a progress bar or not (requires tqdm).
                           Proxies to the file system loader.
            parser: A blob parser which knows how to parse blobs into documents,
                    will instantiate a default parser if not provided.
                    The default can be overridden by either passing a parser or
                    setting the class attribute `blob_parser` (the latter
                    should be used with inheritance).
            parser_kwargs: Keyword arguments to pass to the parser.

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
            if parser == "default":
                try:
                    # If there is an implementation of get_parser on the class, use it.
                    blob_parser = cls.get_parser(**(parser_kwargs or {}))
                except NotImplementedError:
                    # if not then use the global registry.
                    blob_parser = get_parser(parser)
            else:
                blob_parser = get_parser(parser)
        else:
            blob_parser = parser
        return cls(blob_loader, blob_parser)

    @staticmethod
    def get_parser(**kwargs: Any) -> BaseBlobParser:
        """Override this method to associate a default parser with the class."""
        raise NotImplementedError()
