from typing import Iterator, List, Union

import requests
from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader


class BrowserlessLoader(BaseLoader):
    """Load webpages with `Browserless` /content endpoint."""

    def __init__(
        self, api_token: str, urls: Union[str, List[str]], text_content: bool = True
    ):
        """Initialize with API token and the URLs to scrape"""
        self.api_token = api_token
        """Browserless API token."""
        self.urls = urls
        """List of URLs to scrape."""
        self.text_content = text_content

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load Documents from URLs."""

        for url in self.urls:
            if self.text_content:
                response = requests.post(
                    "https://chrome.browserless.io/scrape",
                    params={
                        "token": self.api_token,
                    },
                    json={
                        "url": url,
                        "elements": [
                            {
                                "selector": "body",
                            }
                        ],
                    },
                )
                yield Document(
                    page_content=response.json()["data"][0]["results"][0]["text"],
                    metadata={
                        "source": url,
                    },
                )
            else:
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
