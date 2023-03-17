"""Loader that uses bs4 to load HTML files, enriching metadata with page title."""

import logging
from typing import Dict, List, Union

from bs4 import BeautifulSoup

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__file__)


class BSHTMLLoader(BaseLoader):
    """Loader that uses beautiful soup to parse HTML files."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load HTML document into document objects."""
        with open(self.file_path, "r") as f:
            soup = BeautifulSoup(f, features="lxml")

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
