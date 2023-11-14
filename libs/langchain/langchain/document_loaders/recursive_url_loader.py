from __future__ import annotations

import asyncio
import logging
import re
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utils.html import extract_sub_links

if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger(__name__)


def _metadata_extractor(raw_html: str, url: str) -> dict:
    """Extract metadata from raw html using BeautifulSoup."""
    metadata = {"source": url}

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning(
            "The bs4 package is required for default metadata extraction. "
            "Please install it with `pip install bs4`."
        )
        return metadata
    soup = BeautifulSoup(raw_html, "html.parser")
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", None)
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", None)
    return metadata


class RecursiveUrlLoader(BaseLoader):
    """Load all child links from a URL page.

    **Security Note**: This loader is a crawler that will start crawling
        at a given URL and then expand to crawl child links recursively.

        Web crawlers should generally NOT be deployed with network access
        to any internal servers.

        Control access to who can submit crawling requests and what network access
        the crawler has.

        While crawling, the crawler may encounter malicious URLs that would lead to a
        server-side request forgery (SSRF) attack.

        To mitigate risks, the crawler by default will only load URLs from the same
        domain as the start URL (controlled via prevent_outside named argument).

        This will mitigate the risk of SSRF attacks, but will not eliminate it.

        For example, if crawling a host which hosts several sites:

        https://some_host/alice_site/
        https://some_host/bob_site/

        A malicious URL on Alice's site could cause the crawler to make a malicious
        GET request to an endpoint on Bob's site. Both sites are hosted on the
        same host, so such a request would not be prevented by default.

        See https://python.langchain.com/docs/security
    """

    def __init__(
        self,
        url: str,
        max_depth: Optional[int] = 2,
        use_async: Optional[bool] = None,
        extractor: Optional[Callable[[str], str]] = None,
        metadata_extractor: Optional[Callable[[str, str], str]] = None,
        exclude_dirs: Optional[Sequence[str]] = (),
        timeout: Optional[int] = 10,
        prevent_outside: bool = True,
        link_regex: Union[str, re.Pattern, None] = None,
        headers: Optional[dict] = None,
        check_response_status: bool = False,
    ) -> None:
        """Initialize with URL to crawl and any subdirectories to exclude.

        Args:
            url: The URL to crawl.
            max_depth: The max depth of the recursive loading.
            use_async: Whether to use asynchronous loading.
                If True, this function will not be lazy, but it will still work in the
                expected way, just not lazy.
            extractor: A function to extract document contents from raw html.
                When extract function returns an empty string, the document is
                ignored.
            metadata_extractor: A function to extract metadata from raw html and the
                source url (args in that order). Default extractor will attempt
                to use BeautifulSoup4 to extract the title, description and language
                of the page.
            exclude_dirs: A list of subdirectories to exclude.
            timeout: The timeout for the requests, in the unit of seconds. If None then
                connection will not timeout.
            prevent_outside: If True, prevent loading from urls which are not children
                of the root url.
            link_regex: Regex for extracting sub-links from the raw html of a web page.
            check_response_status: If True, check HTTP response status and skip
                URLs with error responses (400-599).
        """

        self.url = url
        self.max_depth = max_depth if max_depth is not None else 2
        self.use_async = use_async if use_async is not None else False
        self.extractor = extractor if extractor is not None else lambda x: x
        self.metadata_extractor = (
            metadata_extractor
            if metadata_extractor is not None
            else _metadata_extractor
        )
        self.exclude_dirs = exclude_dirs if exclude_dirs is not None else ()

        if any(url.startswith(exclude_dir) for exclude_dir in self.exclude_dirs):
            raise ValueError(
                f"Base url is included in exclude_dirs. Received base_url: {url} and "
                f"exclude_dirs: {self.exclude_dirs}"
            )

        self.timeout = timeout
        self.prevent_outside = prevent_outside if prevent_outside is not None else True
        self.link_regex = link_regex
        self._lock = asyncio.Lock() if self.use_async else None
        self.headers = headers
        self.check_response_status = check_response_status

    def _get_child_links_recursive(
        self, url: str, visited: Set[str], *, depth: int = 0
    ) -> Iterator[Document]:
        """Recursively get all child links starting with the path of the input URL.

        Args:
            url: The URL to crawl.
            visited: A set of visited URLs.
            depth: Current depth of recursion. Stop when depth >= max_depth.
        """

        if depth >= self.max_depth:
            return

        # Get all links that can be accessed from the current URL
        visited.add(url)
        try:
            response = requests.get(url, timeout=self.timeout, headers=self.headers)
            if self.check_response_status and 400 <= response.status_code <= 599:
                raise ValueError(f"Received HTTP status {response.status_code}")
        except Exception as e:
            logger.warning(
                f"Unable to load from {url}. Received error {e} of type "
                f"{e.__class__.__name__}"
            )
            return
        content = self.extractor(response.text)
        if content:
            yield Document(
                page_content=content,
                metadata=self.metadata_extractor(response.text, url),
            )

        # Store the visited links and recursively visit the children
        sub_links = extract_sub_links(
            response.text,
            url,
            base_url=self.url,
            pattern=self.link_regex,
            prevent_outside=self.prevent_outside,
            exclude_prefixes=self.exclude_dirs,
        )
        for link in sub_links:
            # Check all unvisited links
            if link not in visited:
                yield from self._get_child_links_recursive(
                    link, visited, depth=depth + 1
                )

    async def _async_get_child_links_recursive(
        self,
        url: str,
        visited: Set[str],
        *,
        session: Optional[aiohttp.ClientSession] = None,
        depth: int = 0,
    ) -> List[Document]:
        """Recursively get all child links starting with the path of the input URL.

        Args:
            url: The URL to crawl.
            visited: A set of visited URLs.
            depth: To reach the current url, how many pages have been visited.
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "The aiohttp package is required for the RecursiveUrlLoader. "
                "Please install it with `pip install aiohttp`."
            )
        if depth >= self.max_depth:
            return []

        # Disable SSL verification because websites may have invalid SSL certificates,
        # but won't cause any security issues for us.
        close_session = session is None
        session = (
            session
            if session is not None
            else aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=False),
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self.headers,
            )
        )
        async with self._lock:  # type: ignore
            visited.add(url)
        try:
            async with session.get(url) as response:
                text = await response.text()
                if self.check_response_status and 400 <= response.status <= 599:
                    raise ValueError(f"Received HTTP status {response.status}")
        except (aiohttp.client_exceptions.InvalidURL, Exception) as e:
            logger.warning(
                f"Unable to load {url}. Received error {e} of type "
                f"{e.__class__.__name__}"
            )
            if close_session:
                await session.close()
            return []
        results = []
        content = self.extractor(text)
        if content:
            results.append(
                Document(
                    page_content=content,
                    metadata=self.metadata_extractor(text, url),
                )
            )
        if depth < self.max_depth - 1:
            sub_links = extract_sub_links(
                text,
                url,
                base_url=self.url,
                pattern=self.link_regex,
                prevent_outside=self.prevent_outside,
                exclude_prefixes=self.exclude_dirs,
            )

            # Recursively call the function to get the children of the children
            sub_tasks = []
            async with self._lock:  # type: ignore
                to_visit = set(sub_links).difference(visited)
                for link in to_visit:
                    sub_tasks.append(
                        self._async_get_child_links_recursive(
                            link, visited, session=session, depth=depth + 1
                        )
                    )
            next_results = await asyncio.gather(*sub_tasks)
            for sub_result in next_results:
                if isinstance(sub_result, Exception) or sub_result is None:
                    # We don't want to stop the whole process, so just ignore it
                    # Not standard html format or invalid url or 404 may cause this.
                    continue
                # locking not fully working, temporary hack to ensure deduplication
                results += [r for r in sub_result if r not in results]
        if close_session:
            await session.close()
        return results

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load web pages.
        When use_async is True, this function will not be lazy,
        but it will still work in the expected way, just not lazy."""
        visited: Set[str] = set()
        if self.use_async:
            results = asyncio.run(
                self._async_get_child_links_recursive(self.url, visited)
            )
            return iter(results or [])
        else:
            return self._get_child_links_recursive(self.url, visited)

    def load(self) -> List[Document]:
        """Load web pages."""
        return list(self.lazy_load())
