from typing import Iterator, List, Optional, Set
from urllib.parse import urljoin, urldefrag

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class RecursiveUrlLoader(BaseLoader):
    """Loads all child links from a given url."""

    def __init__(
        self,
        url: str,
        exclude_dirs: Optional[str] = None,
        max_depth: int = -1
    ) -> None:
        """Initialize with URL to crawl and any subdirectories to exclude.

        Args:
            url: The URL to crawl.
            exclude_dirs: A list of subdirectories to exclude.
        """

        self.url = url
        self.exclude_dirs = exclude_dirs
        self.max_depth = max_depth

    def get_child_links_recursive(
        self, url: str, depth: int, visited: Optional[Set[str]] = None
    ) -> Iterator[Document]:
        """Recursively get all child links starting with the path of the input URL.

        Args:
            url: The URL to crawl.
            visited: A set of visited URLs.
        """

        from langchain.document_loaders import WebBaseLoader

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "The BeautifulSoup package is required for the RecursiveUrlLoader."
            )

        # Exclude the root and parent from a list
        visited = set() if visited is None else visited

        if self.max_depth > 0 and depth <= self.max_depth:
            return None

        # Exclude the links that start with any of the excluded directories
        if self.exclude_dirs and any(
            url.startswith(exclude_dir) for exclude_dir in self.exclude_dirs
        ):
            return visited

        yield from WebBaseLoader(web_path=url).load()
        visited.add(url)

        # Get all links that are relative to the root of the website
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        all_links = [urljoin(url, link.get("href")) for link in soup.find_all("a")]
        # Filter children url of current url
        child_links = [link for link in set(all_links) if link.startswith(url)]
        # Remove framents to avoid repititions
        defraged_child_links = [urldefrag(link).url for link in child_links]

        # Store the visited links and recursively visit the children
        for link in set(defraged_child_links):
            # Check all unvisited links
            if link not in visited:
                visited.add(link)
                yield from WebBaseLoader(link).load()
                # If the link is a directory (w/ children) then visit it
                if link.endswith("/"):
                    yield from self.get_child_links_recursive(link, depth+1, visited)

        return visited

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load web pages."""
        return self.get_child_links_recursive(self.url, depth=0)

    def load(self) -> List[Document]:
        """Load web pages."""
        return list(self.lazy_load())
