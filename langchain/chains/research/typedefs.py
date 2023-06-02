import abc
from typing import List

from langchain.document_loaders.blob_loaders import Blob


class BlobCrawler(abc.ABC):
    """Crawl a blob and identify links to related content."""

    @abc.abstractmethod
    def crawl(self, blob: Blob, query: str) -> List[str]:
        """Explore the blob and identify links to relevant content."""
