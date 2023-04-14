"""Loading logic for loading documents from a directory."""
import logging
from pathlib import Path
from typing import List, Type, Union, Generator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader, Blob
from langchain.document_loaders.html_bs import BSHTMLLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

FILE_LOADER_TYPE = Union[
    Type[UnstructuredFileLoader], Type[TextLoader], Type[BSHTMLLoader]
]
logger = logging.getLogger(__file__)


def _is_visible(p: Path) -> bool:
    parts = p.parts
    for _p in parts:
        if _p.startswith("."):
            return False
    return True


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
        """Initialize with path to directory and how to glob over it."""
        if loader_kwargs is None:
            loader_kwargs = {}
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.loader_cls = loader_cls
        self.loader_kwargs = loader_kwargs
        self.silent_errors = silent_errors
        self.recursive = recursive

    def load(self) -> List[Document]:
        """Load documents."""
        docs = []

        for blob in self.lazy_load():
            try:
                sub_docs = self.loader_cls(blob.data, **self.loader_kwargs).load()
                docs.extend(sub_docs)
            except Exception as e:
                if self.silent_errors:
                    logger.warning(e)
                else:
                    raise e
        return docs

    def lazy_load(
        self,
    ) -> Union[Generator[Blob, None, None], Generator[Document, None, None]]:
        p = Path(self.path)
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for item in items:
            if item.is_file():
                if _is_visible(item.relative_to(p)) or self.load_hidden:
                    yield Blob.from_path(str(item))
