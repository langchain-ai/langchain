from typing import Iterator, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class RecursiveUrlLoader(BaseLoader):
    """Loads all child links from a given url."""

    def __init__(
        self,
        url: str,
        exclude_dirs: Optional[str] = None,
    ) -> None:
        """Initialize with URL to crawl and any subdirectories to exclude.

        Args:
            url: The URL to crawl.
            exclude_dirs: A list of subdirectories to exclude.
        """

        self.url = url
        self.exclude_dirs = exclude_dirs

    def get_child_links_recursive(
        self, url: str, visited: Optional[Set[str]] = None
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

        # Construct the base and parent URLs
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        parent_url = "/".join(parsed_url.path.split("/")[:-1])
        current_path = parsed_url.path

        # Add a trailing slash if not present
        if not base_url.endswith("/"):
            base_url += "/"
        if not parent_url.endswith("/"):
            parent_url += "/"

        # Exclude the root and parent from a list
        visited = set() if visited is None else visited

        # Exclude the links that start with any of the excluded directories
        if self.exclude_dirs and any(
            url.startswith(exclude_dir) for exclude_dir in self.exclude_dirs
        ):
            return visited

        # Get all links that are relative to the root of the website
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        all_links = [link.get("href") for link in soup.find_all("a")]

        # Extract only the links that are children of the current URL
        child_links = list(
            {
                link
                for link in all_links
                if link and link.startswith(current_path) and link != current_path
            }
        )

        # Get absolute path for all root relative links listed
        absolute_paths = [urljoin(base_url, link) for link in child_links]

        # Store the visited links and recursively visit the children
        for link in absolute_paths:
            # Check all unvisited links
            if link not in visited:
                visited.add(link)
                loaded_link = WebBaseLoader(link).load()
                if isinstance(loaded_link, list):
                    yield from loaded_link
                else:
                    yield loaded_link
                # If the link is a directory (w/ children) then visit it
                if link.endswith("/"):
                    yield from self.get_child_links_recursive(link, visited)

        return visited

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load web pages."""
        return self.get_child_links_recursive(self.url)

    def load(self) -> List[Document]:
        """Load web pages."""
        return list(self.lazy_load())
