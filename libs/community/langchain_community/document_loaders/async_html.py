import asyncio
import logging
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

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


class AsyncHtmlLoader(BaseLoader):
    """Load `HTML` asynchronously."""

    def __init__(
        self,
        web_path: Union[str, List[str]],
        header_template: Optional[dict] = None,
        verify_ssl: Optional[bool] = True,
        proxies: Optional[dict] = None,
        autoset_encoding: bool = True,
        encoding: Optional[str] = None,
        default_parser: str = "html.parser",
        requests_per_second: int = 2,
        requests_kwargs: Optional[Dict[str, Any]] = None,
        raise_for_status: bool = False,
        ignore_load_errors: bool = False,
        *,
        preserve_order: bool = True,
        trust_env: bool = False,
    ):
        """Initialize with a webpage path."""

        # TODO: Deprecate web_path in favor of web_paths, and remove this
        # left like this because there are a number of loaders that expect single
        # urls
        if isinstance(web_path, str):
            self.web_paths = [web_path]
        elif isinstance(web_path, List):
            self.web_paths = web_path

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

        self.session = requests.Session()
        self.session.headers = dict(headers)
        self.session.verify = verify_ssl

        if proxies:
            self.session.proxies.update(proxies)

        self.requests_per_second = requests_per_second
        self.default_parser = default_parser
        self.requests_kwargs = requests_kwargs or {}
        self.raise_for_status = raise_for_status
        self.autoset_encoding = autoset_encoding
        self.encoding = encoding
        self.ignore_load_errors = ignore_load_errors
        self.preserve_order = preserve_order

        self.trust_env = trust_env

    def _fetch_valid_connection_docs(self, url: str) -> Any:
        if self.ignore_load_errors:
            try:
                return self.session.get(url, **self.requests_kwargs)
            except Exception as e:
                warnings.warn(str(e))
                return None

        return self.session.get(url, **self.requests_kwargs)

    @staticmethod
    def _check_parser(parser: str) -> None:
        """Check that parser is valid for bs4."""
        valid_parsers = ["html.parser", "lxml", "xml", "lxml-xml", "html5lib"]
        if parser not in valid_parsers:
            raise ValueError(
                "`parser` must be one of " + ", ".join(valid_parsers) + "."
            )

    async def _fetch(
        self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> str:
        async with aiohttp.ClientSession(trust_env=self.trust_env) as session:
            for i in range(retries):
                try:
                    async with session.get(
                        url,
                        headers=self.session.headers,
                        ssl=None if self.session.verify else False,
                        **self.requests_kwargs,
                    ) as response:
                        try:
                            text = await response.text()
                        except UnicodeDecodeError:
                            logger.error(f"Failed to decode content from {url}")
                            text = ""
                        return text
                except (aiohttp.ClientConnectionError, TimeoutError) as e:
                    if i == retries - 1 and self.ignore_load_errors:
                        logger.warning(f"Error fetching {url} after {retries} retries.")
                        return ""
                    elif i == retries - 1:
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
    ) -> Tuple[str, str]:
        async with semaphore:
            return url, await self._fetch(url)

    async def _lazy_fetch_all(
        self, urls: List[str], preserve_order: bool
    ) -> AsyncIterator[Tuple[str, str]]:
        semaphore = asyncio.Semaphore(self.requests_per_second)
        tasks = [
            asyncio.create_task(self._fetch_with_rate_limit(url, semaphore))
            for url in urls
        ]
        try:
            from tqdm.asyncio import tqdm_asyncio

            if preserve_order:
                for task in tqdm_asyncio(
                    tasks, desc="Fetching pages", ascii=True, mininterval=1
                ):
                    yield await task
            else:
                for task in tqdm_asyncio.as_completed(
                    tasks, desc="Fetching pages", ascii=True, mininterval=1
                ):
                    yield await task
        except ImportError:
            warnings.warn("For better logging of progress, `pip install tqdm`")
            if preserve_order:
                for result in await asyncio.gather(*tasks):
                    yield result
            else:
                for task in asyncio.as_completed(tasks):
                    yield await task

    async def fetch_all(self, urls: List[str]) -> List[str]:
        """Fetch all urls concurrently with rate limiting."""
        return [doc async for _, doc in self._lazy_fetch_all(urls, True)]

    def _to_document(self, url: str, text: str) -> Document:
        from bs4 import BeautifulSoup

        if url.endswith(".xml"):
            parser = "xml"
        else:
            parser = self.default_parser
        self._check_parser(parser)
        soup = BeautifulSoup(text, parser)
        metadata = _build_metadata(soup, url)
        return Document(page_content=text, metadata=metadata)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        results: List[str]
        try:
            # Raises RuntimeError if there is no current event loop.
            asyncio.get_running_loop()
            # If there is a current event loop, we need to run the async code
            # in a separate loop, in a separate thread.
            with ThreadPoolExecutor(max_workers=1) as executor:
                future: Future[List[str]] = executor.submit(
                    asyncio.run,  # type: ignore[arg-type]
                    self.fetch_all(self.web_paths),  # type: ignore[arg-type]
                )
                results = future.result()
        except RuntimeError:
            results = asyncio.run(self.fetch_all(self.web_paths))

        for i, text in enumerate(cast(List[str], results)):
            yield self._to_document(self.web_paths[i], text)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        async for url, text in self._lazy_fetch_all(
            self.web_paths, self.preserve_order
        ):
            yield self._to_document(url, text)
