from typing import Iterator, List, Union

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class BrowserlessLoader(BaseLoader):
    """Loads the content of webpages using Browserless' /content endpoint"""

    def __init__(self, api_token: str, urls: Union[str, List[str]]):
        """Initialize with API token and the URLs to scrape"""
        self.api_token = api_token
        """Browserless API token."""
        self.urls = urls
        """List of URLs to scrape."""

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load Documents from URLs."""

        for url in self.urls:
            response = requests.post(
                "https://chrome.browserless.io/content",
                params={
                    "token": self.api_token,
                },
                json={
                    "url": url,
                },
            )
            yield Document(
                page_content=response.text,
                metadata={
                    "source": url,
                },
            )

    def load(self) -> List[Document]:
        """Load Documents from URLs."""
        return list(self.lazy_load())
