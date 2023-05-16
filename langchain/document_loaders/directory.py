"""Loading logic for loading documents from a directory."""
import concurrent
import logging
from pathlib import Path
from typing import Any, List, Optional, Type, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.html_bs import BSHTMLLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

FILE_LOADER_TYPE = Union[
    Type[UnstructuredFileLoader], Type[TextLoader], Type[BSHTMLLoader]
]
logger = logging.getLogger(__name__)


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
        show_progress: bool = False,
        use_multithreading: bool = False,
        max_concurrency: int = 4,
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
        self.show_progress = show_progress
        self.use_multithreading = use_multithreading
        self.max_concurrency = max_concurrency

    def load_file(
        self, item: Path, path: Path, docs: List[Document], pbar: Optional[Any]
    ) -> None:
        if item.is_file():
            if _is_visible(item.relative_to(path)) or self.load_hidden:
                try:
                    sub_docs = self.loader_cls(str(item), **self.loader_kwargs).load()
                    docs.extend(sub_docs)
                except Exception as e:
                    if self.silent_errors:
                        logger.warning(e)
                    else:
                        raise e
                finally:
                    if pbar:
                        pbar.update(1)

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.path)
        docs: List[Document] = []
        items = list(p.rglob(self.glob) if self.recursive else p.glob(self.glob))

        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(items))
            except ImportError as e:
                logger.warning(
                    "To log the progress of DirectoryLoader you need to install tqdm, "
                    "`pip install tqdm`"
                )
                if self.silent_errors:
                    logger.warning(e)
                else:
                    raise e

        if self.use_multithreading:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrency
            ) as executor:
                executor.map(lambda i: self.load_file(i, p, docs, pbar), items)
        else:
            for i in items:
                self.load_file(i, p, docs, pbar)

        if pbar:
            pbar.close()

        return docs


#
