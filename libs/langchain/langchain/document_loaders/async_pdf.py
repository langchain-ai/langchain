import asyncio
import os
import logging
import requests
import tempfile
import warnings
from typing import Any, List, Union, Optional
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
        self._web_paths = web_paths
        self._requests_per_second = requests_per_second
        self._retries = retries
        self._http_connect_timeout = http_connect_timeout
        self._http_request_timeout = http_request_timeout
        self._verify_ssl = verify_ssl
        self._headers = get_request_headers(header_template)
        self._temp_dirs = []

    def __del__(self) -> None:
        if self._temp_dirs is not None and len(self._temp_dirs) > 0:
            for temp_dir in self._temp_dirs:
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
        self._temp_dirs.append(temp_dir)
        _, suffix = os.path.splitext(url)
        temp_pdf = os.path.join(temp_dir.name, f"tmp{suffix}")

        # Download the PDF.
        with requests.get(
                url,
                stream=True,
                allow_redirects=True,
                headers=self._headers,
                verify=self._verify_ssl,
                timeout=(self._http_connect_timeout, self._http_request_timeout)) as r:
            r.raise_for_status()
            logger.info(f"Status ok downloading '{url}';  Saving to file '{temp_pdf}'")
            with open(temp_pdf, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return PyPDFLoader(temp_pdf)

    async def _fetch_with_rate_limit(
        self, url: str, semaphore: asyncio.Semaphore
    ) -> list[Document]:
        async with semaphore:
            return await self._fetch(url, retries=self._retries)

    async def fetch_all(self, urls: List[str]) -> Any:
        """
        Implementation of fetch_all which allows for individual tasks to fail without
        stopping the entire process.
        :param urls: the urls to fetch.
        :return: list of fetched page contents.
        """
        semaphore = asyncio.Semaphore(self._requests_per_second)
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(self._fetch_with_rate_limit(url, semaphore))
            tasks.append(task)
        try:
            from tqdm.asyncio import tqdm_asyncio

            results = []
            for done in tqdm_asyncio.as_completed(tasks, desc="Fetching PDF documents", ascii=True, mininterval=1):
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

        results = asyncio.run(self.fetch_all(self._web_paths))
        docs = []
        for pdf_pages in results:
            docs.extend(pdf_pages)

        return docs
