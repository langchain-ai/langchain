import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Union, cast

import aiohttp
import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utils.http import get_request_headers

logger = logging.getLogger(__name__)


class AsyncHtmlLoader(BaseLoader):
    """Load `HTML` asynchronously."""

    def __init__(
        self,
        web_path: Union[str, List[str]],
        header_template: Optional[dict] = None,
        verify_ssl: Optional[bool] = True,
        proxies: Optional[dict] = None,
        requests_per_second: int = 2,
        requests_kwargs: Optional[Dict[str, Any]] = None,
        raise_for_status: bool = False,
    ):
        """Initialize with a webpage path."""

        # TODO: Deprecate web_path in favor of web_paths, and remove this
        # left like this because there are a number of loaders that expect single
        # urls
        if isinstance(web_path, str):
            self.web_paths = [web_path]
        elif isinstance(web_path, List):
            self.web_paths = web_path

        headers = get_request_headers(header_template)

        self.session = requests.Session()
        self.session.headers = dict(headers)
        self.session.verify = verify_ssl

        if proxies:
            self.session.proxies.update(proxies)

        self.requests_per_second = requests_per_second
        self.requests_kwargs = requests_kwargs or {}
        self.raise_for_status = raise_for_status

    async def _fetch(
        self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> str:
        async with aiohttp.ClientSession() as session:
            for i in range(retries):
                try:
                    async with session.get(
                        url,
                        headers=self.session.headers,
                        ssl=None if self.session.verify else False,
                    ) as response:
                        try:
                            text = await response.text()
                        except UnicodeDecodeError:
                            logger.error(f"Failed to decode content from {url}")
                            text = ""
                        return text
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
        """
        Fetch all urls concurrently with rate limiting.  Exceptions are handled and logged, since
        we don't want a single url failure to stop the entire process.
        :param urls: the urls to fetch.
        :return: list of fetched page contents.
        """
        semaphore = asyncio.Semaphore(self.requests_per_second)
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(self._fetch_with_rate_limit(url, semaphore))
            tasks.append(task)
        try:
            from tqdm.asyncio import tqdm_asyncio

            results = []
            for done in tqdm_asyncio.as_completed(tasks, desc="Fetching pages", ascii=True, mininterval=1):
                try:
                    results.append(await done)
                except Exception as e:
                    logger.warning(f"Exception while fetching page: {e}")
            return results

        except ImportError:
            warnings.warn("For better logging of progress, `pip install tqdm`")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for exception in [r for r in results if isinstance(r, Exception)]:
                logger.warning(f"Exception while fetching page: {exception}")

            return [r for r in results if not isinstance(r, Exception)]

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        for doc in self.load():
            yield doc

    def load(self) -> List[Document]:
        """Load text from the url(s) in web_path."""

        try:
            # Raises RuntimeError if there is no current event loop.
            asyncio.get_running_loop()
            # If there is a current event loop, we need to run the async code
            # in a separate loop, in a separate thread.
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.fetch_all(self.web_paths))
                results = future.result()
        except RuntimeError:
            results = asyncio.run(self.fetch_all(self.web_paths))
        docs = []
        for i, text in enumerate(cast(List[str], results)):
            metadata = {"source": self.web_paths[i]}
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
