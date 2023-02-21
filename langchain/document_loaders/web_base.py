"""Web base loader class."""
from typing import Any, List

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class WebBaseLoader(BaseLoader):
    """Loader that uses urllib and beautiful soup to load webpages."""

    def __init__(self, web_path: str):
        """Initialize with webpage path."""
        self.web_path = web_path

    @staticmethod
    def _scrape(url: str) -> Any:
        from bs4 import BeautifulSoup

        html_doc = requests.get(url)
        soup = BeautifulSoup(html_doc.text, "html.parser")
        return soup

    def scrape(self) -> Any:
        """Scrape data from webpage and return it in BeautifulSoup format."""
        return self._scrape(self.web_path)

    def load(self) -> List[Document]:
        """Load data into document objects."""
        soup = self.scrape()
        text = soup.get_text()
        metadata = {"source": self.web_path}
        return [Document(page_content=text, metadata=metadata)]
