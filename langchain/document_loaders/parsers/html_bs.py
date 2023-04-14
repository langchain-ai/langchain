"""Loader that uses bs4 to load HTML files, enriching metadata with page title."""

import logging
from typing import Dict, Union, Generator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders.schema import Blob

logger = logging.getLogger(__file__)


class BSHTMLParser(BaseBlobParser):
    """Loader that uses beautiful soup to parse HTML files."""

    def __init__(
        self,
        bs_kwargs: Union[dict, None] = None,
    ) -> None:
        """Initialise with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object."""
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ValueError(
                "bs4 package not found, please install it with `pip install bs4`"
            )

        if bs_kwargs is None:
            bs_kwargs = {"features": "lxml"}
        self.bs_kwargs = bs_kwargs

    def lazy_parse(self, blob: Blob) -> Generator[Document, None, None]:
        """Load HTML document into document objects."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(blob.as_string(), **self.bs_kwargs)
        text = soup.get_text()

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        metadata: Dict[str, Union[str, None]] = {
            "source": blob.path,
            "title": title,
        }
        yield Document(page_content=text, metadata=metadata)
