import logging
from pathlib import Path
from typing import List, Type, Union, Generator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader, Blob
from langchain.document_loaders.html_bs import BSHTMLLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

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
        recursive: bool = False,
    ):
        """Initialize with path to directory and how to glob over it."""
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.silent_errors = silent_errors
        self.recursive = recursive

    def load(self) -> List[Document]:
        raise NotImplementedError("Do not use.")

    def lazy_load(
        self,
    ) -> Union[Generator[Blob, None, None], Generator[Document, None, None]]:
        p = Path(self.path)
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for item in items:
            if item.is_file():
                if _is_visible(item.relative_to(p)) or self.load_hidden:
                    yield Blob.from_path(str(item))
