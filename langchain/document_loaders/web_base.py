"""Web base loader class."""
import asyncio
import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Union

import aiohttp
import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)

default_header_template = {
    "User-Agent": "",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
    ";q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _build_metadata(soup: Any, url: str) -> dict:
    """Build metadata from BeautifulSoup output."""
    metadata = {"source": url}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", None)
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", None)
    return metadata


class WebBaseLoader(BaseLoader):
    """Loader that uses urllib and beautiful soup to load webpages."""

    web_paths: List[str]

    requests_per_second: int = 2
    """Max number of concurrent requests to make."""

    default_parser: str = "html.parser"
    """Default parser to use for BeautifulSoup."""

    requests_kwargs: Dict[str, Any] = {}
    """kwargs for requests"""

    raise_for_status: bool = False
    """Raise an exception if http status code denotes an error."""

    bs_get_text_kwargs: Dict[str, Any] = {}
    """kwargs for beatifulsoup4 get_text"""

    def __init__(
        self,
        web_path: Union[str, List[str]],
        header_template: Optional[dict] = None,
        verify: Optional[bool] = True,
        proxies: Optional[dict] = None,
    ):
        """Initialize with webpage path."""

        # TODO: Deprecate web_path in favor of web_paths, and remove this
        # left like this because there are a number of loaders that expect single
        # urls
        if isinstance(web_path, str):
            self.web_paths = [web_path]
        elif isinstance(web_path, List):
            self.web_paths = web_path

        self.session = requests.Session()
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ValueError(
                "bs4 package not found, please install it with " "`pip install bs4`"
            )

        # Choose to verify
        self.verify = verify

        headers = header_template or default_header_template
        if not headers.get("User-Agent"):
            try:
                from fake_useragent import UserAgent

                headers["User-Agent"] = UserAgent().random
            except ImportError:
                logger.info(
                    "fake_useragent not found, using default user agent."
                    "To get a realistic header for requests, "
                    "`pip install fake_useragent`."
                )
        self.session.headers = dict(headers)

        if proxies:
            self.session.proxies.update(proxies)

    @property
    def web_path(self) -> str:
        if len(self.web_paths) > 1:
            raise ValueError("Multiple webpaths found.")
        return self.web_paths[0]

    async def _fetch(
        self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> str:
        # For SiteMap SSL verification
        if not self.requests_kwargs.get("verify", True):
            connector = aiohttp.TCPConnector(ssl=False)
        else:
            connector = None

        async with aiohttp.ClientSession(connector=connector) as session:
            for i in range(retries):
                try:
                    async with session.get(
                        url, headers=self.session.headers, verify=self.verify
                    ) as response:
                        return await response.text()
                except aiohttp.ClientConnectionError as e:
                    if i == retries - 1:
                        raise
                    else:
                        logger.warning(
                            f"Error fetching {url} with attempt "
                            f"{i + 1}/{retries}: {e}. Retrying..."
                        )
                        await asyncio.sleep(cooldown * backoff**i)
        raise ValueError("retry count exceeded")

    async def _fetch_with_rate_limit(
        self, url: str, semaphore: asyncio.Semaphore
    ) -> str:
        async with semaphore:
            return await self._fetch(url)

    async def fetch_all(self, urls: List[str]) -> Any:
        """Fetch all urls concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(self.requests_per_second)
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(self._fetch_with_rate_limit(url, semaphore))
            tasks.append(task)
        try:
            from tqdm.asyncio import tqdm_asyncio

            return await tqdm_asyncio.gather(
                *tasks, desc="Fetching pages", ascii=True, mininterval=1
            )
        except ImportError:
            warnings.warn("For better logging of progress, `pip install tqdm`")
            return await asyncio.gather(*tasks)

    @staticmethod
    def _check_parser(parser: str) -> None:
        """Check that parser is valid for bs4."""
        valid_parsers = ["html.parser", "lxml", "xml", "lxml-xml", "html5lib"]
        if parser not in valid_parsers:
            raise ValueError(
                "`parser` must be one of " + ", ".join(valid_parsers) + "."
            )

    def scrape_all(self, urls: List[str], parser: Union[str, None] = None) -> List[Any]:
        """Fetch all urls, then return soups for all results."""
        from bs4 import BeautifulSoup

        results = asyncio.run(self.fetch_all(urls))
        final_results = []
        for i, result in enumerate(results):
            url = urls[i]
            if parser is None:
                if url.endswith(".xml"):
                    parser = "xml"
                else:
                    parser = self.default_parser
                self._check_parser(parser)
            final_results.append(BeautifulSoup(result, parser))

        return final_results

    def _scrape(self, url: str, parser: Union[str, None] = None) -> Any:
        from bs4 import BeautifulSoup

        if parser is None:
            if url.endswith(".xml"):
                parser = "xml"
            else:
                parser = self.default_parser

        self._check_parser(parser)

        html_doc = self.session.get(url, verify=self.verify, **self.requests_kwargs)
        if self.raise_for_status:
            html_doc.raise_for_status()
        html_doc.encoding = html_doc.apparent_encoding
        return BeautifulSoup(html_doc.text, parser)

    def scrape(self, parser: Union[str, None] = None) -> Any:
        """Scrape data from webpage and return it in BeautifulSoup format."""

        if parser is None:
            parser = self.default_parser

        return self._scrape(self.web_path, parser)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        for path in self.web_paths:
            soup = self._scrape(path)
            text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = _build_metadata(soup, path)
            yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        """Load text from the url(s) in web_path."""
        return list(self.lazy_load())

    def aload(self) -> List[Document]:
        """Load text from the urls in web_path async into Documents."""

        results = self.scrape_all(self.web_paths)
        docs = []
        for i in range(len(results)):
            soup = results[i]
            text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = _build_metadata(soup, self.web_paths[i])
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
