"""Loader that uses bs4 to load HTML files, enriching metadata with page title."""

import logging
from typing import Dict, List, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


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

        self.file_path = file_path
        self.open_encoding = open_encoding
        if bs_kwargs is None:
            bs_kwargs = {"features": "lxml"}
        self.bs_kwargs = bs_kwargs

    def load(self) -> List[Document]:
        from bs4 import BeautifulSoup

        """Load HTML document into document objects."""
        with open(self.file_path, "r", encoding=self.open_encoding) as f:
            soup = BeautifulSoup(f, **self.bs_kwargs)

        text = soup.get_text()

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        metadata: Dict[str, Union[str, None]] = {
            "source": self.file_path,
            "title": title,
        }
        return [Document(page_content=text, metadata=metadata)]
