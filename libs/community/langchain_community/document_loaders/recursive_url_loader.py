from __future__ import annotations

import asyncio
import inspect
import logging
import re
from typing import (
    AsyncIterator,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import aiohttp
import requests
from langchain_core.documents import Document
from langchain_core.utils.html import extract_sub_links

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


def _metadata_extractor(
    raw_html: str, url: str, response: Union[requests.Response, aiohttp.ClientResponse]
) -> dict:
    """Extract metadata from raw html using BeautifulSoup."""
    content_type = getattr(response, "headers").get("Content-Type", "")
    metadata = {"source": url, "content_type": content_type}

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
        # TODO: Deprecate use_async
        use_async: Optional[bool] = None,
        extractor: Optional[Callable[[str], str]] = None,
        metadata_extractor: Optional[_MetadataExtractorType] = None,
        exclude_dirs: Optional[Sequence[str]] = (),
        timeout: Optional[int] = 10,
        prevent_outside: bool = True,
        link_regex: Union[str, re.Pattern, None] = None,
        headers: Optional[dict] = None,
        check_response_status: bool = False,
        continue_on_failure: bool = True,
        *,
        base_url: Optional[str] = None,
        autoset_encoding: bool = True,
        encoding: Optional[str] = None,
        retries: int = 0,
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
            metadata_extractor: A function to extract metadata from args: raw html, the
                source url, and the requests.Response/aiohttp.ClientResponse object
                (args in that order).
                Default extractor will attempt to use BeautifulSoup4 to extract the
                title, description and language of the page.
                ..code-block:: python

                    import requests
                    import aiohttp

                    def simple_metadata_extractor(
                        raw_html: str, url: str, response: Union[requests.Response, aiohttp.ClientResponse]
                    ) -> dict:
                        content_type = getattr(response, "headers").get("Content-Type", "")
                        return {"source": url, "content_type": content_type}

            exclude_dirs: A list of subdirectories to exclude.
            timeout: The timeout for the requests, in the unit of seconds. If None then
                connection will not timeout.
            prevent_outside: If True, prevent loading from urls which are not children
                of the root url.
            link_regex: Regex for extracting sub-links from the raw html of a web page.
            headers: Default request headers to use for all requests.
            check_response_status: If True, check HTTP response status and skip
                URLs with error responses (400-599).
            continue_on_failure: If True, continue if getting or parsing a link raises
                an exception. Otherwise, raise the exception.
            base_url: The base url to check for outside links against.
            autoset_encoding: Whether to automatically set the encoding of the response.
                If True, the encoding of the response will be set to the apparent
                encoding, unless the `encoding` argument has already been explicitly set.
            encoding: The encoding of the response. If manually set, the encoding will be
                set to given value, regardless of the `autoset_encoding` argument.
        """  # noqa: E501

        if exclude_dirs and any(url.startswith(dir) for dir in exclude_dirs):
            raise ValueError(
                f"Base url is included in exclude_dirs. Received base_url: {url} and "
                f"exclude_dirs: {exclude_dirs}"
            )
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be positive.")

        self.url = url
        self.max_depth = max_depth if max_depth is not None else 2
        self.use_async = use_async if use_async is not None else False
        self.extractor = extractor if extractor is not None else lambda x: x
        if metadata_extractor is None:
            self.metadata_extractor = _metadata_extractor
        else:
            self.metadata_extractor = _wrap_metadata_extractor(metadata_extractor)
        self.autoset_encoding = autoset_encoding
        self.encoding = encoding
        self.exclude_dirs = exclude_dirs if exclude_dirs is not None else ()
        self.timeout = timeout
        self.prevent_outside = prevent_outside if prevent_outside is not None else True
        self.link_regex = link_regex
        self.headers = headers
        self.check_response_status = check_response_status
        self.continue_on_failure = continue_on_failure
        self.base_url = base_url if base_url is not None else url
        self.retries = retries

    def _lazy_load_recursive(
        self, url: str, visited: Set[str], *, depth: int = 0
    ) -> Iterator[Document]:
        if url in visited:
            raise ValueError
        visited.add(url)
        response = None
        for _ in range(self.retries + 1):
            response = self._request(url)
            if response:
                break
        if not response:
            return
        text = response.text
        if content := self.extractor(text):
            metadata = self.metadata_extractor(text, url, response)
            yield Document(content, metadata=metadata)

        if depth + 1 < self.max_depth:
            for link in self._extract_sub_links(text, url):
                if link not in visited:
                    yield from self._lazy_load_recursive(link, visited, depth=depth + 1)
                if link not in visited:
                    raise ValueError

    async def _async_get_child_links_recursive(
        self,
        url: str,
        visited: Set[str],
        session: aiohttp.ClientSession,
        *,
        depth: int = 0,
    ) -> AsyncIterator[Document]:
        if depth >= self.max_depth:
            return

        visited.add(url)
        try:
            async with session.get(url) as response:
                if self.check_response_status:
                    response.raise_for_status()
                text = await response.text()
        except (aiohttp.client_exceptions.InvalidURL, Exception) as e:
            if self.continue_on_failure:
                logger.warning(
                    f"Unable to load {url}. Received error {e} of type "
                    f"{e.__class__.__name__}"
                )
                return
            else:
                raise e

        if content := self.extractor(text):
            metadata = self.metadata_extractor(text, url, response)
            yield Document(content, metadata=metadata)
        for link in self._extract_sub_links(text, url):
            if link not in visited:
                async for doc in self._async_get_child_links_recursive(
                    link, visited, session, depth=depth + 1
                ):
                    yield doc

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load web pages.
        When use_async is True, this function will not be lazy,
        but it will still work in the expected way, just not lazy."""
        if self.use_async:

            async def aload():
                results = []
                async for doc in self.alazy_load():
                    results.append(doc)
                return results

            return iter(asyncio.run(aload()))

        else:
            yield from self._lazy_load_recursive(self.url, set())

    async def alazy_load(self) -> AsyncIterator[Document]:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=False),
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self.headers,
        ) as session:
            async for doc in self._async_get_child_links_recursive(
                self.url, set(), session
            ):
                yield doc

    def _extract_sub_links(self, html: str, url: str) -> List[str]:
        return extract_sub_links(
            html,
            url,
            base_url=self.base_url,
            pattern=self.link_regex,
            prevent_outside=self.prevent_outside,
            exclude_prefixes=self.exclude_dirs,
            continue_on_failure=self.continue_on_failure,
        )

    def _request(self, url: str) -> Optional[requests.Response]:
        try:
            response = requests.get(url, timeout=self.timeout, headers=self.headers)

            if self.encoding is not None:
                response.encoding = self.encoding
            elif self.autoset_encoding:
                response.encoding = response.apparent_encoding
            else:
                pass

            if self.check_response_status:
                response.raise_for_status()
        except Exception as e:
            if self.continue_on_failure:
                logger.warning(
                    f"Unable to load {url}. Received error {e} of type "
                    f"{e.__class__.__name__}"
                )
                return None
            else:
                raise e
        else:
            return response


_MetadataExtractorType1 = Callable[[str, str], dict]
_MetadataExtractorType2 = Callable[
    [str, str, Union[requests.Response, aiohttp.ClientResponse]], dict
]
_MetadataExtractorType = Union[_MetadataExtractorType1, _MetadataExtractorType2]


def _wrap_metadata_extractor(
    metadata_extractor: _MetadataExtractorType,
) -> _MetadataExtractorType2:
    if len(inspect.signature(metadata_extractor).parameters) == 3:
        return cast(_MetadataExtractorType2, metadata_extractor)
    else:

        def _metadata_extractor_wrapper(
            raw_html: str,
            url: str,
            response: Union[requests.Response, aiohttp.ClientResponse],
        ) -> dict:
            return cast(_MetadataExtractorType1, metadata_extractor)(raw_html, url)

        return _metadata_extractor_wrapper
