from typing import Iterator, List, Optional, Set, Callable, Union, Literal
from urllib.parse import urlparse

import re

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

try:
    import asyncio
except ImportError:
    raise ImportError(
        "The asyncio package is required for the RecursiveUrlLoader. "
        "Please install it with `pip install asyncio`."
    )
try: 
    import aiohttp
except ImportError:
    raise ImportError(
        "The aiohttp package is required for the RecursiveUrlLoader. "
        "Please install it with `pip install aiohttp`."
    )

class AsyncRecursiveUrlLoader(BaseLoader):
    """Loads all child links from a given url."""

    def __init__(
        self,
        url: str,
        exclude_dirs: Optional[str] = None,
        raw_webpage_to_text_converter: Callable[[str], str] = lambda raw: raw,
        max_depth: int = 2,
        prevent_outside: bool = False,
        timeout: aiohttp.ClientTimeout = aiohttp.ClientTimeout(100),
    ) -> None:
        """Initialize with URL to crawl and any subdirectories to exclude.

        Args:
            url: The URL to crawl.
            exclude_dirs: A list of subdirectories to exclude.
            raw_webpage_to_text_converter: A function that converts raw webpages to the text of the generated document. It is recommended to use other tools to extract important infos.
            max_depth: If to reach the page would need to go through more pages than max_depth, then stop.
            prevent_outside: If the parameter is true, websites out of the url won't be crawled.
            timeout: timeout for the aiohttp session, use aiohttp.ClientTimeout.
        """

        self.url = url
        self.exclude_dirs = exclude_dirs
        self.raw_webpage_to_text_converter = raw_webpage_to_text_converter
        self.max_depth = max_depth
        self.prevent_outside = prevent_outside
        self.timeout = timeout

    async def get_child_links_recursive(
        self, url: str, visited: Optional[Set[str]] = None, depth: int = 0
    ) -> List[Document]:
        """Recursively get all child links starting with the path of the input URL.

        Args:
            url: The URL to crawl.
            visited: A set of visited URLs.
            depth: To reach the current url, how many pages have been visited.
        """
        
        if depth > self.max_depth:
            return

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
            return
        # Disable SSL verification because some websites may have invalid SSL certificates, but won't cause any security issues for us.
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False), timeout=self.timeout) as session:
            # Some url may be invalid, so catch the exception
            response: aiohttp.ClientResponse
            try:
                response = await session.get(url)
                text = await response.text()
            except aiohttp.client_exceptions.InvalidURL:
                return
            # There may be some other exceptions, so catch them, we don't want to stop the whole process
            except Exception as e:
                return
            
            # Get all links that are relative to the root of the website
            all_links = re.findall(r"href=[\"\'](.*?)[\"\']", text)
            
            absolute_paths = []
            # Process the links
            for link in all_links:
                # Here are blacklist patterns
                # Exclude the links that start with javascript: or mailto:
                if link.startswith("javascript:") or link.startswith("mailto:"):
                    continue
                
                # Blacklist patterns end.
                
                # Here are whitelist patterns
                
                # Some links may be in form of /path/to/link, so add the base URL
                if link.startswith("/") and not link.startswith("//"):
                    absolute_paths.append(base_url + link[1:])
                # Some links may be in form of //path/to/link, so add the scheme
                elif link.startswith("//"):
                    absolute_paths.append(parsed_url.scheme + ":" + link)
                # Extract only the links that are children of the current URL
                # Only the links without the previous two patterns are possible links to outside.
                elif link.startswith(current_path) and link != current_path:
                    absolute_paths.append(link)
                    
                # Whitelist patterns end.
                
                # Despite prevent outside should be blacklist rule, it must be done here or it could filter valid ones.
                elif (not self.prevent_outside) or link.startswith(current_path):
                    pass
                
                # Some links may be in form of path/to/link, so add the parent URL
                else:
                    absolute_paths.append(parent_url + link)

            # Worker will be only called within the current function to act as an async generator
            # Worker function will process the link and then recursively call get_child_links_recursive to process the children
            async def worker(
                link: str
            ) -> Union[Document, None]:
                try:
                    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False), timeout=self.timeout) as session:
                        response = await session.get(link)
                        text = await response.text()
                        extracted = self.raw_webpage_to_text_converter(text)
                        if len(extracted) > 0:
                            return Document(
                                page_content=extracted,
                                metadata={
                                    "source": link,
                                }
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
            results = list(filter(lambda x: x is not None, await asyncio.gather(*tasks)))
            # Recursively call the function to get the children of the children
            sub_tasks = []
            for link in absolute_paths:
                sub_tasks.append(self.get_child_links_recursive(link, visited, depth + 1))
            # sub_tasks returns coroutines of list, so we need to flatten the list await asyncio.gather(*sub_tasks)
            flattened = []
            for sub_result in await asyncio.gather(*sub_tasks):
                if sub_result is not None:
                    flattened += sub_result
            results += flattened
            return list(filter(lambda x: x is not None, results))

    def lazy_load(self) -> Iterator[Document]:
        """Actually, because the crawler is async, it is not lazy."""
        results = asyncio.run(self.get_child_links_recursive(self.url))
        # Yield the results
        if results is None:
            results = []
        for result in results:
            yield result

    def load(self) -> List[Document]:
        """Load web pages."""
        return list(self.lazy_load())