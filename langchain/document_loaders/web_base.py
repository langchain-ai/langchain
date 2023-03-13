"""Web base loader class."""
import random
from typing import Any, List

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

header_template = {
    "User-Agent": "",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


class WebBaseLoader(BaseLoader):
    """Loader that uses urllib and beautiful soup to load webpages."""

    def __init__(self, web_path: str):
        """Initialize with webpage path."""
        self.web_path = web_path
        self.session = requests.Session()

        try:
            from fake_useragent import UserAgent

            header_template["User-Agent"] = UserAgent().random
            self.session.headers = dict(header_template)
        except ImportError:
            print(
                "fake_useragent not found, using default user agent."
                "To get a realistic header for requests, `pip install fake_useragent`."
            )

    def _scrape(self, url: str) -> Any:
        from bs4 import BeautifulSoup

        html_doc = self.session.get(url)
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
