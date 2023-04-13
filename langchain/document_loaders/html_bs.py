"""Loader that uses bs4 to load HTML files, enriching metadata with page title."""

import logging
from typing import List, Union, Generator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader, Blob
from langchain.document_loaders.parsers.html_bs import BSHTMLParser

logger = logging.getLogger(__file__)


class BSHTMLLoader(BaseLoader):
    """Loader that uses beautiful soup to parse HTML files."""

    def __init__(
        self,
        file_path: str,
        open_encoding: Union[str, None] = None,
        bs_kwargs: Union[dict, None] = None,
    ) -> None:
        """Initialise with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object."""
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ValueError(
                "bs4 package not found, please install it with " "`pip install bs4`"
            )

        self.parser = BSHTMLParser(bs_kwargs=bs_kwargs)
        self.file_path = file_path
        self.open_encoding = open_encoding

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Generator[Document, None, None]:
        """Lazy load"""
        blob = Blob.from_file(self.file_path)
        yield from self.parser.lazy_parse(blob)
