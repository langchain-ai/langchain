"""Loading logic for loading documents from a directory."""
import logging
from typing import List, Type, Union, Generator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.blob_loaders.file_system import FileSystemLoader
from langchain.document_loaders.html_bs import BSHTMLLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

FILE_LOADER_TYPE = Union[
    Type[UnstructuredFileLoader], Type[TextLoader], Type[BSHTMLLoader]
]

logger = logging.getLogger(__file__)


class DirectoryLoader(BaseLoader):
    """Loading logic for loading documents from a directory."""

    def __init__(
        self,
        path: str,
        glob: str = "**/[!.]*",
        silent_errors: bool = False,
        load_hidden: bool = False,
        loader_cls: FILE_LOADER_TYPE = UnstructuredFileLoader,
        loader_kwargs: Union[dict, None] = None,
        recursive: bool = False,
    ):
        """Initialize with path to directory and how to glob over it.

        Args:
            path: Path to directory.
            glob: Glob pattern to use.
            silent_errors: If True, errors will be logged and ignored.
            load_hidden: If True, hidden files will be loaded.
            loader_cls: Class to use for loading files.
            loader_kwargs: Keyword arguments to pass to loader_cls.
            recursive: If True, will recursively load files.
        """
        self.loader = FileSystemLoader(
            path, glob, load_hidden=load_hidden, recursive=recursive
        )
        self.loader_cls = loader_cls
        self.loader_kwargs = loader_kwargs or {}
        self.silent_errors = silent_errors

    def load(self) -> List[Document]:
        """Load documents."""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Generator[Document, None, None]:
        """Load documents lazily."""
        for blob in self.loader.yield_blobs():
            try:
                sub_docs = self.loader_cls(blob.data, **self.loader_kwargs).load()
                yield from sub_docs
            except Exception as e:
                if self.silent_errors:
                    logger.warning(e)
                else:
                    raise e
