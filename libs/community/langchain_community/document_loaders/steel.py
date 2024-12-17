import logging
from typing import Iterator, List, Optional, Union

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)

class SteelLoader(BaseLoader):
    """Load webpages with Steel."""

    def __init__(
        self, api_token: str, urls: Union[str, List[str]], text_content: bool = True
    ):
        """Initialize with API token and the URLs to scrape"""
        self.api_token = api_token
        """Steel API token."""
        self.urls = urls
        """List of URLs to scrape."""
        self.text_content = text_content

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load Documents from URLs."""

        for url in self.urls:
            try:
                if self.text_content:
                    response = requests.post(
                        "https://api.steel.dev/scrape",
                        headers={
                            "Authorization": f"Bearer {self.api_token}",
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
                    response.raise_for_status()
                    yield Document(
                        page_content=response.json()["data"][0]["results"][0]["text"],
                        metadata={
                            "source": url,
                        },
                    )
                else:
                    response = requests.post(
                        "https://api.steel.dev/content",
                        headers={
                            "Authorization": f"Bearer {self.api_token}",
                        },
                        json={
                            "url": url,
                        },
                    )
                    response.raise_for_status()
                    yield Document(
                        page_content=response.text,
                        metadata={
                            "source": url,
                        },
                    )
            except requests.RequestException as e:
                logger.error(f"Error fetching {url}: {e}")
                continue
