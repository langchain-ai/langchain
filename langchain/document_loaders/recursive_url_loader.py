import re
from typing import Callable, Iterator, List, Optional, Set, Union
from urllib.parse import urljoin, urlparse

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

try:
    import aiohttp
except ImportError:
    print("The aiohttp package is required for the RecursiveUrlLoader.")
    print("Please install it with `pip install aiohttp`.")

try:
    import asyncio
except ImportError:
    print("The asyncio package is required for the RecursiveUrlLoader.")
    print("Please install it with `pip install asyncio`.")

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("The bs4 package is required for the RecursiveUrlLoader.")
    print("Please install it with `pip install bs4`.")


class RecursiveUrlLoader(BaseLoader):
    """Loads all child links from a given url."""

    def __init__(
        self,
        url: str,
        exclude_dirs: Optional[str] = None,
        use_async: bool = False,
        extractor: Callable[[str], str] = lambda x: x,
        max_depth: int = 2,
        timeout: int = 10,
        prevent_outside: bool = True,
    ) -> None:
        """Initialize with URL to crawl and any subdirectories to exclude.

        Args:
            url: The URL to crawl.
            exclude_dirs: A list of subdirectories to exclude.
            use_async: Whether to use asynchronous loading, if use async, lazy load will not be lazy, despite it will work as usual.
            extractor: A function to extract the text from the html.
            max_depth: The max depth of the recursive loading.
            timeout: The timeout for the requests, in the unit of seconds.
        """

        self.url = url
        self.exclude_dirs = exclude_dirs
        self.use_async = use_async
        self.extractor = extractor
        self.max_depth = max_depth
        self.timeout = timeout
        self.prevent_outside = prevent_outside

    def _get_sub_links(self, raw_html: str, base_url: str) -> List[str]:
        """This function extracts all the links from the raw html, and convert them into absolute paths.

        Args:
            raw_html (str): original html
            base_url (str): the base url of the html

        Returns:
            List[str]: sub links
        """
        # Get all links that are relative to the root of the website
        all_links = re.findall(r"href=[\"\'](.*?)[\"\']", raw_html)

        absolute_paths = []
        # Process the links
        for link in all_links:
            # Some patterns are on blacklist
            # like javascript: or mailto:
            if link.startswith("javascript:") or link.startswith("mailto:"):
                continue
            # Or #, which is a link to the same page
            if link.startswith("#"):
                continue

            # Some may be absolute links like https://to/path
            if link.startswith("http"):
                if (not self.prevent_outside) or (
                    self.prevent_outside and link.startswith(base_url)
                ):
                    absolute_paths.append(link)
            else:
                absolute_paths.append(urljoin(base_url, link))

            # Some may be relative links like /to/path
            if link.startswith("/") and not link.startswith("//"):
                absolute_paths.append(urljoin(base_url, link))
                continue
            # Some may have omitted the protocol like //to/path
            if link.startswith("//"):
                absolute_paths.append(f"{urlparse(base_url).scheme}:{link}")
                continue

        # Remove duplicates, also do a filter to prevent outside links
        absolute_paths = list(
            set(
                filter(
                    lambda x: not self.prevent_outside
                    or x.startswith(base_url)
                    and (x != base_url),
                    absolute_paths,
                )
            )
        )

        return absolute_paths

    def _gen_metadata(self, raw_html: str, url: str) -> dict:
        """Build metadata from BeautifulSoup output."""
        metadata = {"source": url}
        soup = BeautifulSoup(raw_html, "html.parser")
        if title := soup.find("title"):
            metadata["title"] = title.get_text()
        if description := soup.find("meta", attrs={"name": "description"}):
            metadata["description"] = description.get("content", None)
        if html := soup.find("html"):
            metadata["language"] = html.get("lang", None)
        return metadata

    def _get_child_links_recursive(
        self, url: str, visited: Optional[Set[str]] = None, depth: int = 0
    ) -> Iterator[Document]:
        """Recursively get all child links starting with the path of the input URL.

        Args:
            url: The URL to crawl.
            visited: A set of visited URLs.
        """

        if depth > self.max_depth:
            return []

        # Add a trailing slash if not present
        if not url.endswith("/"):
            url += "/"

        # Exclude the root and parent from a list
        visited = set() if visited is None else visited

        # Exclude the links that start with any of the excluded directories
        if self.exclude_dirs and any(
            url.startswith(exclude_dir) for exclude_dir in self.exclude_dirs
        ):
            return []

        # Get all links that can be accessed from the current URL
        try:
            response = requests.get(url, timeout=self.timeout)
        except:
            return []

        absolute_paths = self._get_sub_links(response.text, url)

        # Store the visited links and recursively visit the children
        for link in absolute_paths:
            # Check all unvisited links
            if link not in visited:
                visited.add(link)
                try:
                    response = requests.get(link)
                    text = response.text
                except:
                    continue
                loaded_link = Document(
                    page_content=self.extractor(text),
                    metadata=self._gen_metadata(text, link),
                )
                yield loaded_link
                # If the link is a directory (w/ children) then visit it
                if link.endswith("/"):
                    yield from self._get_child_links_recursive(link, visited, depth + 1)

        return []

    async def async_get_child_links_recursive(
        self, url: str, visited: Optional[Set[str]] = None, depth: int = 0
    ) -> List[Document]:
        """Recursively get all child links starting with the path of the input URL.

        Args:
            url: The URL to crawl.
            visited: A set of visited URLs.
            depth: To reach the current url, how many pages have been visited.
        """

        if depth > self.max_depth:
            return []

        # Construct the base and parent URLs
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        # Add a trailing slash if not present
        if not base_url.endswith("/"):
            base_url += "/"

        # Exclude the root and parent from a list
        visited = set() if visited is None else visited

        # Exclude the links that start with any of the excluded directories
        if self.exclude_dirs and any(
            url.startswith(exclude_dir) for exclude_dir in self.exclude_dirs
        ):
            return []
        # Disable SSL verification because some websites may have invalid SSL certificates, but won't cause any security issues for us.
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(verify_ssl=False),
            timeout=aiohttp.ClientTimeout(self.timeout),
        ) as session:
            # Some url may be invalid, so catch the exception
            response: aiohttp.ClientResponse
            try:
                response = await session.get(url)
                text = await response.text()
            except aiohttp.client_exceptions.InvalidURL:
                return []
            # There may be some other exceptions, so catch them, we don't want to stop the whole process
            except Exception as e:
                return []

            absolute_paths = self._get_sub_links(text, base_url)

            # Worker will be only called within the current function to act as an async generator
            # Worker function will process the link and then recursively call get_child_links_recursive to process the children
            async def worker(link: str) -> Union[Document, None]:
                try:
                    async with aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(verify_ssl=False),
                        timeout=aiohttp.ClientTimeout(self.timeout),
                    ) as session:
                        response = await session.get(link)
                        text = await response.text()
                        extracted = self.extractor(text)
                        if len(extracted) > 0:
                            return Document(
                                page_content=extracted,
                                metadata=self._gen_metadata(text, link),
                            )
                        else:
                            return None
                # Despite the fact that we have filtered some links, there may still be some invalid links, so catch the exception
                except aiohttp.client_exceptions.InvalidURL:
                    return None
                # There may be some other exceptions, so catch them, we don't want to stop the whole process
                except Exception as e:
                    # print(e)
                    return None

            # The coroutines that will be executed
            tasks = []
            # Generate the tasks
            for link in absolute_paths:
                # Check all unvisited links
                if link not in visited:
                    visited.add(link)
                    tasks.append(worker(link))
            # Get the not None results
            results = list(
                filter(lambda x: x is not None, await asyncio.gather(*tasks))
            )
            # Recursively call the function to get the children of the children
            sub_tasks = []
            for link in absolute_paths:
                sub_tasks.append(
                    self._get_child_links_recursive(link, visited, depth + 1)
                )
            # sub_tasks returns coroutines of list, so we need to flatten the list await asyncio.gather(*sub_tasks)
            flattened = []
            next_results = await asyncio.gather(*sub_tasks)
            for sub_result in next_results:
                if sub_result is not None:
                    flattened += sub_result
            results += flattened
            return list(filter(lambda x: x is not None, results))

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load web pages."""
        if self.use_async:
            results = asyncio.run(self.async_get_child_links_recursive(self.url))
            if results is None:
                return iter([])
            else:
                return iter(results)
        else:
            return self._get_child_links_recursive(self.url)

    def load(self) -> List[Document]:
        """Load web pages."""
        return list(self.lazy_load())
