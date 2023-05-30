"""Module contains code for fetching documents from the web using playwright.

This module currently re-uses the code from the `web_base` module to avoid
re-implementing rate limiting behavior.

The module contains downloading interfaces.

Sub-classing with the given interface should allow a user to add url based
user-agents and authentication if needed.

Downloading is batched by default to allow parallelizing efficiently.
"""

import abc
import asyncio
import mimetypes
from bs4 import BeautifulSoup
from typing import Sequence, List, Any

from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders.blob_loaders import Blob


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
    def download(self, urls: Sequence[str]) -> List[Blob]:
        """Download a url synchronously."""
        raise NotImplementedError()

    async def adownload(self, urls: Sequence[str]) -> List[Blob]:
        """Download a url asynchronously."""
        raise NotImplementedError()


class PlaywrightDownloadHandler(DownloadHandler):
    """Download URLS using playwright.

    This is an implementation of the download handler that uses playwright to download
    urls. This is useful for downloading urls that require javascript to be executed.
    """

    def download(self, urls: Sequence[str]) -> List[Blob]:
        """Download list of urls synchronously."""
        # Implement using a threadpool or using playwright API if it supports it
        raise NotImplementedError()

    async def _download(self, browser: Any, url: str) -> str:
        """Download a url asynchronously using playwright."""
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        html_content = await page.content()
        return html_content

    async def adownload(self, urls: Sequence[str]) -> List[Blob]:
        """Download a url asynchronously using playwright.

        Args:
            url: The url to download.

        Returns:
            The html content of the url.
        """
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            tasks = [self._download(browser, url) for url in urls]
            contents = await asyncio.gather(*tasks, return_exceptions=True)
            await browser.close()

        return _repackage_as_blobs(urls, contents)


class RequestsDownloadHandler(DownloadHandler):
    def __init__(self, web_downloader: WebBaseLoader):
        self.web_downloader = web_downloader

    def download(self, url: str) -> str:
        """Download a url synchronously."""
        # Implement with threadpool.
        raise NotImplementedError()

    async def adownload(self, urls: Sequence[str]) -> List[Blob]:
        """Download a batch of urls asynchronously using playwright."""
        download = WebBaseLoader(web_path=[])  # Place holder
        contents = await download.fetch_all(list(urls))
        return _repackage_as_blobs(urls, contents)


def _repackage_as_blobs(urls: Sequence[str], contents: Sequence[str]) -> List[Blob]:
    """Repackage the contents as blobs."""
    return [
        Blob(data=content, mimetype=mimetypes.guess_type(url))
        for url, content in zip(urls, contents)
    ]


class AutoDownloadHandler(DownloadHandler):
    def __init__(self, web_downloader: WebBaseLoader) -> None:
        """Initialize the auto download handler."""
        self.requests_downloader = RequestsDownloadHandler(web_downloader)
        self.playwright_downloader = PlaywrightDownloadHandler()

    async def adownload(self, urls: Sequence[str]) -> List[Blob]:
        """Download a batch of urls asynchronously using playwright."""
        # Check if javascript is required
        blobs = await self.requests_downloader.adownload(urls)

        # Check if javascript is required
        must_redownload = [
            (idx, url)
            for idx, (url, blob) in enumerate(zip(urls, blobs))
            if _is_javascript_required(blob.data)
        ]

        indexes, urls_to_redownload = zip(*must_redownload)
        contents = await self.playwright_downloader.adownload(urls)
        return contents
