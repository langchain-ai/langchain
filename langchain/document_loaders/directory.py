"""Loading logic for loading documents from a directory."""
import logging
from pathlib import Path
from typing import List, Type, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

FILE_LOADER_TYPE = Union[Type[UnstructuredFileLoader], Type[TextLoader]]
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
        recursive: bool = False,
    ):
        """Initialize with path to directory and how to glob over it."""
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.loader_cls = loader_cls
        self.silent_errors = silent_errors
        self.recursive = recursive

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.path)
        docs = []
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for i in items:
            if i.is_file():
                if _is_visible(i.relative_to(p)) or self.load_hidden:
                    try:
                        sub_docs = self.loader_cls(str(i)).load()
                        docs.extend(sub_docs)
                    except Exception as e:
                        if self.silent_errors:
                            logger.warning(e)
                        else:
                            raise e
        return docs
