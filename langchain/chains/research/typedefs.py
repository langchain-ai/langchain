from typing import List, Sequence, Mapping, Any

import abc

from langchain.document_loaders.blob_loaders import Blob


class AbstractQueryGenerator(abc.ABC):
    """Abstract class for generating queries."""

    @abc.abstractmethod
    def generate_queries(self, question: str) -> List[str]:
        """Generate queries for the given question."""
        raise NotImplementedError()


class AbstractSearcher(abc.ABC):
    """Abstract class for running searches."""

    def search(self, queries: Sequence[str]) -> List[Mapping[str, Any]]:
        """Run a search for the given query.

        Args:
            queries: the query to run the search for.

        Returns:
            a list of search results.
        """
        raise NotImplementedError()


class BlobCrawler(abc.ABC):
    """Crawl a blob and identify links to related content."""

    @abc.abstractmethod
    def crawl(self, blob: Blob, query: str) -> List[str]:
        """Explore the blob and identify links to related content that is relevant to the query."""
