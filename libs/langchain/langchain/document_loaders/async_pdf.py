import asyncio
import logging
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Union, Optional

import requests

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.base import BaseLoader
from langchain.utils.http import get_request_headers

logger = logging.getLogger(__name__)


class AsyncPdfLoader(BaseLoader):
    """
    This class is used to load pdf files asynchronously.
    """

    def __init__(
        self,
        web_paths: list[str],
        requests_per_second: Optional[int] = 2,
        retries: Optional[int] = 3,
        http_connect_timeout: Optional[int] = 30,
        http_request_timeout: Optional[int] = 300,
        verify_ssl: Optional[Union[bool, str]] = True,
        header_template: Optional[dict] = None,
    ):
        self.web_paths = web_paths
        self.requests_per_second = requests_per_second
        self.retries = retries
        self.http_connect_timeout = http_connect_timeout
        self.http_request_timeout = http_request_timeout
        self.verify_ssl = verify_ssl
        self.headers = get_request_headers(header_template)
        self.temp_dirs = []

    def __del__(self) -> None:
        if self.temp_dirs is not None and len(self.temp_dirs) > 0:
            for temp_dir in self.temp_dirs:
                temp_dir.cleanup()

    async def _fetch(
        self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> list[Document]:
        for i in range(retries):
            try:
                pdf = self._download_pdf(url)

                pages = pdf.load_and_split()
                for n, page in enumerate(pages):
                    page.metadata["source"] = f"{url} (page {n})"

                return pages
            except Exception as e:
                if i == retries - 1:
                    raise
                else:
                    logger.warning(
                        f"Error fetching {url} with attempt "
                        f"{i + 1}/{retries}: {e}. Retrying..."
                    )
                    await asyncio.sleep(cooldown * backoff**i)

        raise ValueError("retry count exceeded")

    def _download_pdf(self, url: str) -> PyPDFLoader:

        # Resolve temp filename for the downloaded PDF.
        temp_dir = tempfile.TemporaryDirectory()
        self.temp_dirs.append(temp_dir)
        temp_pdf = os.path.join(temp_dir.name, f"tmp.pdf")

        # Download the PDF.
        with requests.get(
                url,
                stream=True,
                allow_redirects=True,
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=(self.http_connect_timeout, self.http_request_timeout)) as r:
            r.raise_for_status()
            logger.info(f"Status ok downloading '{url}'; Saving to file '{temp_pdf}'")
            with open(temp_pdf, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return PyPDFLoader(temp_pdf)

    async def _fetch_with_rate_limit(
        self, url: str, semaphore: asyncio.Semaphore
    ) -> list[Document]:
        async with semaphore:
            return await self._fetch(url, retries=self.retries)

    async def fetch_all(self, urls: List[str]) -> Any:
        """
        Implementation of fetch_all which allows for individual tasks to fail without
        stopping the entire process.
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
            for done in tqdm_asyncio.as_completed(
                    tasks,
                    desc="Fetching PDF documents",
                    ascii=True,
                    mininterval=1
            ):
                try:
                    results.append(await done)
                except Exception as e:
                    logger.warning(f"Exception while fetching PDF: {e}")
            return results

        except ImportError:
            warnings.warn("For better logging of progress, `pip install tqdm`")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for exception in [r for r in results if isinstance(r, Exception)]:
                logger.warning(f"Exception while fetching PDF: {exception}")

            return [r for r in results if not isinstance(r, Exception)]

    def load(self) -> List[Document]:
        """
        Load PDF from the url(s) in web_path.
        """
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
        for pdf_pages in results:
            docs.extend(pdf_pages)

        return docs
