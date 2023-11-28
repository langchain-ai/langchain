import logging
from typing import Any, List

import requests
from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class DiffbotLoader(BaseLoader):
    """Load `Diffbot` json file."""

    def __init__(
        self, api_token: str, urls: List[str], continue_on_failure: bool = True
    ):
        """Initialize with API token, ids, and key.

        Args:
            api_token: Diffbot API token.
            urls: List of URLs to load.
            continue_on_failure: Whether to continue loading other URLs if one fails.
               Defaults to True.
        """
        self.api_token = api_token
        self.urls = urls
        self.continue_on_failure = continue_on_failure

    def _diffbot_api_url(self, diffbot_api: str) -> str:
        return f"https://api.diffbot.com/v3/{diffbot_api}"

    def _get_diffbot_data(self, url: str) -> Any:
        """Get Diffbot file from Diffbot REST API."""
        # TODO: Add support for other Diffbot APIs
        diffbot_url = self._diffbot_api_url("article")
        params = {
            "token": self.api_token,
            "url": url,
        }
        response = requests.get(diffbot_url, params=params, timeout=10)

        # TODO: handle non-ok errors
        return response.json() if response.ok else {}

    def load(self) -> List[Document]:
        """Extract text from Diffbot on all the URLs and return Documents"""
        docs: List[Document] = list()

        for url in self.urls:
            try:
                data = self._get_diffbot_data(url)
                text = data["objects"][0]["text"] if "objects" in data else ""
                metadata = {"source": url}
                docs.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching or processing {url}, exception: {e}")
                else:
                    raise e
        return docs
