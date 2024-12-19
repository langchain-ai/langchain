"""Web base loader class."""

import asyncio
import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import aiohttp
import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utils.user_agent import get_user_agent

logger = logging.getLogger(__name__)

default_header_template = {
    "User-Agent": get_user_agent(),
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
        metadata["description"] = description.get("content", "No description found.")
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", "No language found.")
    return metadata


class WebBaseLoader(BaseLoader):
    """
    WebBaseLoader document loader integration

    Setup:
        Install ``langchain_community``.

        .. code-block:: bash

            pip install -U langchain_community

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import WebBaseLoader

            loader = WebBaseLoader(
                web_path = "https://www.espn.com/"
                # header_template = None,
                # verify_ssl = True,
                # proxies = None,
                # continue_on_failure = False,
                # autoset_encoding = True,
                # encoding = None,
                # web_paths = (),
                # requests_per_second = 2,
                # default_parser = "html.parser",
                # requests_kwargs = None,
                # raise_for_status = False,
                # bs_get_text_kwargs = None,
                # bs_kwargs = None,
                # session = None,
                # show_progress = True,
                # trust_env = False,
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

            ESPN - Serving Sports Fans. Anytime. Anywhere.

            {'source': 'https://www.espn.com/', 'title': 'ESPN - Serving Sports Fans. Anytime. Anywhere.', 'description': 'Visit ESPN for live scores, highlights and sports news. Stream exclusive games on ESPN+ and play fantasy sports.', 'language': 'en'}


    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            ESPN - Serving Sports Fans. Anytime. Anywhere.

            {'source': 'https://www.espn.com/', 'title': 'ESPN - Serving Sports Fans. Anytime. Anywhere.', 'description': 'Visit ESPN for live scores, highlights and sports news. Stream exclusive games on ESPN+ and play fantasy sports.', 'language': 'en'}

    """  # noqa: E501

    def __init__(
        self,
        web_path: Union[str, Sequence[str]] = "",
        header_template: Optional[dict] = None,
        verify_ssl: bool = True,
        proxies: Optional[dict] = None,
        continue_on_failure: bool = False,
        autoset_encoding: bool = True,
        encoding: Optional[str] = None,
        web_paths: Sequence[str] = (),
        requests_per_second: int = 2,
        default_parser: str = "html.parser",
        requests_kwargs: Optional[Dict[str, Any]] = None,
        raise_for_status: bool = False,
        bs_get_text_kwargs: Optional[Dict[str, Any]] = None,
        bs_kwargs: Optional[Dict[str, Any]] = None,
        session: Any = None,
        *,
        show_progress: bool = True,
        trust_env: bool = False,
    ) -> None:
        """Initialize loader.

        Args:
            web_paths: Web paths to load from.
            requests_per_second: Max number of concurrent requests to make.
            default_parser: Default parser to use for BeautifulSoup.
            requests_kwargs: kwargs for requests
            raise_for_status: Raise an exception if http status code denotes an error.
            bs_get_text_kwargs: kwargs for beatifulsoup4 get_text
            bs_kwargs: kwargs for beatifulsoup4 web page parsing
            show_progress: Show progress bar when loading pages.
            trust_env: set to True if using proxy to make web requests, for example
                using http(s)_proxy environment variables. Defaults to False.
        """
        # web_path kept for backwards-compatibility.
        if web_path and web_paths:
            raise ValueError(
                "Received web_path and web_paths. Only one can be specified. "
                "web_path is deprecated, web_paths should be used."
            )
        if web_paths:
            self.web_paths = list(web_paths)
        elif isinstance(web_path, str):
            self.web_paths = [web_path]
        elif isinstance(web_path, Sequence):
            self.web_paths = list(web_path)
        else:
            raise TypeError(
                f"web_path must be str or Sequence[str] got ({type(web_path)}) or"
                f" web_paths must be Sequence[str] got ({type(web_paths)})"
            )
        self.requests_per_second = requests_per_second
        self.default_parser = default_parser
        self.requests_kwargs = requests_kwargs or {}
        self.raise_for_status = raise_for_status
        self.show_progress = show_progress
        self.bs_get_text_kwargs = bs_get_text_kwargs or {}
        self.bs_kwargs = bs_kwargs or {}
        if session:
            self.session = session
        else:
            session = requests.Session()
            header_template = header_template or default_header_template.copy()
            if not header_template.get("User-Agent"):
                try:
                    from fake_useragent import UserAgent

                    header_template["User-Agent"] = UserAgent().random
                except ImportError:
                    logger.info(
                        "fake_useragent not found, using default user agent."
                        "To get a realistic header for requests, "
                        "`pip install fake_useragent`."
                    )
            session.headers = dict(header_template)
            session.verify = verify_ssl
            if proxies:
                session.proxies.update(proxies)
            self.session = session
        self.continue_on_failure = continue_on_failure
        self.autoset_encoding = autoset_encoding
        self.encoding = encoding
        self.trust_env = trust_env

    @property
    def web_path(self) -> str:
        if len(self.web_paths) > 1:
            raise ValueError("Multiple webpaths found.")
        return self.web_paths[0]

    async def _fetch(
        self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> str:
        async with aiohttp.ClientSession(trust_env=self.trust_env) as session:
            for i in range(retries):
                try:
                    kwargs: Dict = dict(
                        headers=self.session.headers,
                        cookies=self.session.cookies.get_dict(),
                    )
                    if not self.session.verify:
                        kwargs["ssl"] = False

                    async with session.get(
                        url, **(self.requests_kwargs | kwargs)
                    ) as response:
                        if self.raise_for_status:
                            response.raise_for_status()
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
            try:
                return await self._fetch(url)
            except Exception as e:
                if self.continue_on_failure:
                    logger.warning(
                        f"Error fetching {url}, skipping due to"
                        f" continue_on_failure=True"
                    )
                    return ""
                logger.exception(
                    f"Error fetching {url} and aborting, use continue_on_failure=True "
                    "to continue loading urls after encountering an error."
                )
                raise e

    async def fetch_all(self, urls: List[str]) -> Any:
        """Fetch all urls concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(self.requests_per_second)
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(self._fetch_with_rate_limit(url, semaphore))
            tasks.append(task)
        try:
            if self.show_progress:
                from tqdm.asyncio import tqdm_asyncio

                return await tqdm_asyncio.gather(
                    *tasks, desc="Fetching pages", ascii=True, mininterval=1
                )
            else:
                return await asyncio.gather(*tasks)
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
            final_results.append(BeautifulSoup(result, parser, **self.bs_kwargs))

        return final_results

    def _scrape(
        self,
        url: str,
        parser: Union[str, None] = None,
        bs_kwargs: Optional[dict] = None,
    ) -> Any:
        from bs4 import BeautifulSoup

        if parser is None:
            if url.endswith(".xml"):
                parser = "xml"
            else:
                parser = self.default_parser

        self._check_parser(parser)

        html_doc = self.session.get(url, **self.requests_kwargs)
        if self.raise_for_status:
            html_doc.raise_for_status()

        if self.encoding is not None:
            html_doc.encoding = self.encoding
        elif self.autoset_encoding:
            html_doc.encoding = html_doc.apparent_encoding
        return BeautifulSoup(html_doc.text, parser, **(bs_kwargs or {}))

    def scrape(self, parser: Union[str, None] = None) -> Any:
        """Scrape data from webpage and return it in BeautifulSoup format."""

        if parser is None:
            parser = self.default_parser

        return self._scrape(self.web_path, parser=parser, bs_kwargs=self.bs_kwargs)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        for path in self.web_paths:
            soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
            text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = _build_metadata(soup, path)
            yield Document(page_content=text, metadata=metadata)

    def aload(self) -> List[Document]:  # type: ignore
        """Load text from the urls in web_path async into Documents."""

        results = self.scrape_all(self.web_paths)
        docs = []
        for path, soup in zip(self.web_paths, results):
            text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = _build_metadata(soup, path)
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
