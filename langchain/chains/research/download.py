"""Module contains code for fetching documents from the web using playwright.

This module currently re-uses the code from the `web_base` module to avoid
re-implementing rate limiting behavior.

The module contains downloading interfaces.

Sub-classing with the given interface should allow a user to add url based
user-agents and authentication if needed.

Downloading is batched by default to allow efficient parallelization.
"""

import abc
import asyncio
import mimetypes
from typing import Any, List, Optional, Sequence

from bs4 import BeautifulSoup

from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders.blob_loaders import Blob

MaybeBlob = Optional[Blob]


def _is_javascript_required(html_content: str) -> bool:
    """Heuristic to determine whether javascript execution is required.

    Args:
        html_content (str): The HTML content to check.

    Returns:
        bool: True if javascript execution is required, False otherwise.
    """
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, "lxml")

    # Count the number of HTML elements
    body = soup.body
    if not body:
        return True
    num_elements = len(body.find_all())
    requires_javascript = num_elements < 1
    return requires_javascript


class DownloadHandler(abc.ABC):
    def download(self, urls: Sequence[str]) -> List[MaybeBlob]:
        """Download a batch of URLs synchronously."""
        raise NotImplementedError()

    async def adownload(self, urls: Sequence[str]) -> List[MaybeBlob]:
        """Download a batch of URLs asynchronously."""
        raise NotImplementedError()


class PlaywrightDownloadHandler(DownloadHandler):
    """Download URLS using playwright.

    This is an implementation of the download handler that uses playwright to download
    urls. This is useful for downloading urls that require javascript to be executed.
    """

    def __init__(self, timeout: int = 5) -> None:
        """Initialize the download handler.

        Args:
            timeout: The timeout in seconds to wait for a page to load.
        """
        self.timeout = timeout

    def download(self, urls: Sequence[str]) -> List[MaybeBlob]:
        """Download list of urls synchronously."""
        return asyncio.run(self.adownload(urls))

    async def _download(self, browser: Any, url: str) -> Optional[str]:
        """Download a url asynchronously using playwright."""
        from playwright.async_api import TimeoutError

        page = await browser.new_page()
        try:
            # Up to 5 seconds to load the page.
            await page.goto(url, wait_until="networkidle")
            html_content = await page.content()
        except TimeoutError:
            html_content = None
        return html_content

    async def adownload(self, urls: Sequence[str]) -> List[MaybeBlob]:
        """Download a batch of URLs asynchronously using playwright.

        Args:
            urls: The urls to download.

        Returns:
            list of blobs containing the downloaded content.
        """
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            tasks = [self._download(browser, url) for url in urls]
            contents = await asyncio.gather(*tasks, return_exceptions=True)
            await browser.close()

        return _repackage_as_blobs(urls, contents)


class RequestsDownloadHandler(DownloadHandler):
    def __init__(self, web_downloader: Optional[WebBaseLoader] = None) -> None:
        """Initialize the requests download handler."""
        self.web_downloader = web_downloader or WebBaseLoader(web_path=[])

    def download(self, urls: Sequence[str]) -> List[MaybeBlob]:
        """Download a batch of URLS synchronously."""
        return asyncio.run(self.adownload(urls))

    async def adownload(self, urls: Sequence[str]) -> List[MaybeBlob]:
        """Download a batch of urls asynchronously using playwright."""
        download = WebBaseLoader(web_path=[])  # Place holder
        contents = await download.fetch_all(list(urls))
        return _repackage_as_blobs(urls, contents)


def _repackage_as_blobs(
    urls: Sequence[str], contents: Sequence[Optional[str]]
) -> List[MaybeBlob]:
    """Repackage the contents as blobs."""
    blobs: List[MaybeBlob] = []
    for url, content in zip(urls, contents):
        mimetype = mimetypes.guess_type(url)[0]
        if content is None:
            blobs.append(None)
        else:
            blobs.append(Blob(data=content or "", mimetype=mimetype, path=url))

    return blobs


class AutoDownloadHandler(DownloadHandler):
    """Download URLs using the requests library if possible.

    Fallback to using playwright if javascript is required.
    """

    def __init__(self, web_downloader: Optional[WebBaseLoader] = None) -> None:
        """Initialize the auto download handler."""
        self.requests_downloader = RequestsDownloadHandler(
            web_downloader or WebBaseLoader(web_path=[])
        )
        self.playwright_downloader = PlaywrightDownloadHandler()

    async def adownload(self, urls: Sequence[str]) -> List[MaybeBlob]:
        """Download a batch of urls asynchronously using playwright."""
        # Check if javascript is required
        blobs = await self.requests_downloader.adownload(urls)

        # Check if javascript is required
        must_redownload = [
            (idx, url)
            for idx, (url, blob) in enumerate(zip(urls, blobs))
            if blob is not None and _is_javascript_required(blob.as_string())
        ]
        if must_redownload:
            indexes, urls_to_redownload = zip(*must_redownload)
            new_blobs = await self.playwright_downloader.adownload(urls_to_redownload)

            for idx, blob in zip(indexes, new_blobs):
                blobs[idx] = blob
        return blobs

    def download(self, urls: Sequence[str]) -> List[MaybeBlob]:
        """Download a batch of URLs synchronously."""
        return asyncio.run(self.adownload(urls))
