"""Web base loader class."""
from typing import List

import requests
from bs4 import BeautifulSoup

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class WebBaseLoader(BaseLoader):
    """Loader that uses urllib and beautiful soup to load webpages."""

    def __init__(self, web_path: str):
        """Initialize with webpage path."""
        self.web_path = web_path

    def scrape(self) -> BeautifulSoup:
        """Scrape data from webpage and return it in BeautifulSoup format."""
        html_doc = requests.get(self.web_path)
        soup = BeautifulSoup(html_doc.text, "html.parser")
        return soup

    def load(self) -> List[Document]:
        """Load data into document objects."""
        soup = self.scrape()
        text = soup.get_text()
        metadata = {"source": self.web_path}
        return [Document(page_content=text, metadata=metadata)]
