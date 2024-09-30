from __future__ import annotations

import asyncio
import inspect
import logging
import re
from typing import (
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
            "Please install it with `pip install -U beautifulsoup4`."
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
    """Recursively load all child links from a root URL.

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

        See https://python.langchain.com/v0.2/docs/security/

    Setup:

        This class has no required additional dependencies. You can optionally install
        ``beautifulsoup4`` for richer default metadata extraction:

        .. code-block:: bash

            pip install -U beautifulsoup4

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import RecursiveUrlLoader

            loader = RecursiveUrlLoader(
                "https://docs.python.org/3.9/",
                # max_depth=2,
                # use_async=False,
                # extractor=None,
                # metadata_extractor=None,
                # exclude_dirs=(),
                # timeout=10,
                # check_response_status=True,
                # continue_on_failure=True,
                # prevent_outside=True,
                # base_url=None,
                # ...
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            <!DOCTYPE html>

            <html xmlns="http://www.w3.org/1999/xhtml">
            <head>
                <meta charset="utf-8" /><
            {'source': 'https://docs.python.org/3.9/', 'content_type': 'text/html', 'title': '3.9.19 Documentation', 'language': None}

    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            <!DOCTYPE html>

            <html xmlns="http://www.w3.org/1999/xhtml">
            <head>
                <meta charset="utf-8" /><
            {'source': 'https://docs.python.org/3.9/', 'content_type': 'text/html', 'title': '3.9.19 Documentation', 'language': None}

    Content parsing / extraction:
        By default the loader sets the raw HTML from each link as the Document page
        content. To parse this HTML into a more human/LLM-friendly format you can pass
        in a custom ``extractor`` method:

            .. code-block:: python

                # This example uses `beautifulsoup4` and `lxml`
                import re
                from bs4 import BeautifulSoup

                def bs4_extractor(html: str) -> str:
                    soup = BeautifulSoup(html, "lxml")
                    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

                loader = RecursiveUrlLoader(
                    "https://docs.python.org/3.9/",
                    extractor=bs4_extractor,
                )
                print(loader.load()[0].page_content[:200])


            .. code-block:: python

                3.9.19 Documentation

                Download
                Download these documents
                Docs by version

                Python 3.13 (in development)
                Python 3.12 (stable)
                Python 3.11 (security-fixes)
                Python 3.10 (security-fixes)
                Python 3.9 (securit

    Metadata extraction:
        Similarly to content extraction, you can specify a metadata extraction function
        to customize how Document metadata is extracted from the HTTP response.

        .. code-block:: python

            import aiohttp
            import requests
            from typing import Union

            def simple_metadata_extractor(
                raw_html: str, url: str, response: Union[requests.Response, aiohttp.ClientResponse]
            ) -> dict:
                content_type = getattr(response, "headers").get("Content-Type", "")
                return {"source": url, "content_type": content_type}

            loader = RecursiveUrlLoader(
                "https://docs.python.org/3.9/",
                metadata_extractor=simple_metadata_extractor,
            )
            loader.load()[0].metadata

        .. code-block:: python

            {'source': 'https://docs.python.org/3.9/', 'content_type': 'text/html'}

    Filtering URLs:
        You may not always want to pull every URL from a website. There are four parameters
        that allow us to control what URLs we pull recursively. First, we can set the
        ``prevent_outside`` parameter to prevent URLs outside of the ``base_url`` from
        being pulled. Note that the ``base_url`` does not need to be the same as the URL we
        pass in, as shown below. We can also use ``link_regex`` and ``exclude_dirs`` to be
        more specific with the URLs that we select. In this example, we only pull websites
        from the python docs, which contain the string "index" somewhere and are not
        located in the FAQ section of the website.

        .. code-block:: python

            loader = RecursiveUrlLoader(
                "https://docs.python.org/3.9/",
                prevent_outside=True,
                base_url="https://docs.python.org",
                link_regex=r'<a\\s+(?:[^>]*?\\s+)?href="([^"]*(?=index)[^"]*)"',
                exclude_dirs=['https://docs.python.org/3.9/faq']
            )
            docs = loader.load()

        .. code-block:: python

            ['https://docs.python.org/3.9/',
            'https://docs.python.org/3.9/py-modindex.html',
            'https://docs.python.org/3.9/genindex.html',
            'https://docs.python.org/3.9/tutorial/index.html',
            'https://docs.python.org/3.9/using/index.html',
            'https://docs.python.org/3.9/extending/index.html',
            'https://docs.python.org/3.9/installing/index.html',
            'https://docs.python.org/3.9/library/index.html',
            'https://docs.python.org/3.9/c-api/index.html',
            'https://docs.python.org/3.9/howto/index.html',
            'https://docs.python.org/3.9/distributing/index.html',
            'https://docs.python.org/3.9/reference/index.html',
            'https://docs.python.org/3.9/whatsnew/index.html']

    """  # noqa: E501

    def __init__(
        self,
        url: str,
        max_depth: Optional[int] = 2,
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
    ) -> None:
        """Initialize with URL to crawl and any subdirectories to exclude.

        Args:
            url: The URL to crawl.
            max_depth: The max depth of the recursive loading.
            use_async: Whether to use asynchronous loading.
                If True, lazy_load function will not be lazy, but it will still work in the
                expected way, just not lazy.
            extractor: A function to extract document contents from raw HTML.
                When extract function returns an empty string, the document is
                ignored. Default returns the raw HTML.
            metadata_extractor: A function to extract metadata from args: raw HTML, the
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

        self.url = url
        self.max_depth = max_depth if max_depth is not None else 2
        self.use_async = use_async if use_async is not None else False
        self.extractor = extractor if extractor is not None else lambda x: x
        metadata_extractor = (
            metadata_extractor
            if metadata_extractor is not None
            else _metadata_extractor
        )
        self.autoset_encoding = autoset_encoding
        self.encoding = encoding
        self.metadata_extractor = _wrap_metadata_extractor(metadata_extractor)
        self.exclude_dirs = exclude_dirs if exclude_dirs is not None else ()

        if any(url.startswith(exclude_dir) for exclude_dir in self.exclude_dirs):
            raise ValueError(
                f"Base url is included in exclude_dirs. Received base_url: {url} and "
                f"exclude_dirs: {self.exclude_dirs}"
            )

        self.timeout = timeout
        self.prevent_outside = prevent_outside if prevent_outside is not None else True
        self.link_regex = link_regex
        self.headers = headers
        self.check_response_status = check_response_status
        self.continue_on_failure = continue_on_failure
        self.base_url = base_url if base_url is not None else url

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

            if self.encoding is not None:
                response.encoding = self.encoding
            elif self.autoset_encoding:
                response.encoding = response.apparent_encoding

            if self.check_response_status and 400 <= response.status_code <= 599:
                raise ValueError(f"Received HTTP status {response.status_code}")
        except Exception as e:
            if self.continue_on_failure:
                logger.warning(
                    f"Unable to load from {url}. Received error {e} of type "
                    f"{e.__class__.__name__}"
                )
                return
            else:
                raise e
        content = self.extractor(response.text)
        if content:
            yield Document(
                page_content=content,
                metadata=self.metadata_extractor(response.text, url, response),
            )

        # Store the visited links and recursively visit the children
        sub_links = extract_sub_links(
            response.text,
            url,
            base_url=self.base_url,
            pattern=self.link_regex,
            prevent_outside=self.prevent_outside,
            exclude_prefixes=self.exclude_dirs,
            continue_on_failure=self.continue_on_failure,
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
        if not self.use_async:
            raise ValueError(
                "Async functions forbidden when not initialized with `use_async`"
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
        visited.add(url)
        try:
            async with session.get(url) as response:
                text = await response.text()
                if self.check_response_status and 400 <= response.status <= 599:
                    raise ValueError(f"Received HTTP status {response.status}")
        except (aiohttp.client_exceptions.InvalidURL, Exception) as e:
            if close_session:
                await session.close()
            if self.continue_on_failure:
                logger.warning(
                    f"Unable to load {url}. Received error {e} of type "
                    f"{e.__class__.__name__}"
                )
                return []
            else:
                raise e
        results = []
        content = self.extractor(text)
        if content:
            results.append(
                Document(
                    page_content=content,
                    metadata=self.metadata_extractor(text, url, response),
                )
            )
        if depth < self.max_depth - 1:
            sub_links = extract_sub_links(
                text,
                url,
                base_url=self.base_url,
                pattern=self.link_regex,
                prevent_outside=self.prevent_outside,
                exclude_prefixes=self.exclude_dirs,
                continue_on_failure=self.continue_on_failure,
            )

            # Recursively call the function to get the children of the children
            sub_tasks = []
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
