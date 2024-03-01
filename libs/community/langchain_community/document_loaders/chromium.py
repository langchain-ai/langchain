import asyncio
import logging
from typing import Iterator, List

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class AsyncChromiumLoader(BaseLoader):
    """Scrape HTML pages from URLs using a
    headless instance of the Chromium."""

    def __init__(
        self,
        urls: List[str],
    ):
        """
        Initialize the loader with a list of URL paths.

        Args:
            urls (List[str]): A list of URLs to scrape content from.

        Raises:
            ImportError: If the required 'playwright' package is not installed.
        """
        self.urls = urls

        try:
            import playwright  # noqa: F401
        except ImportError:
            raise ImportError(
                "playwright is required for AsyncChromiumLoader. "
                "Please install it with `pip install playwright`."
            )

    async def ascrape_playwright(self, url: str) -> str:
        """
        Asynchronously scrape the content of a given URL using Playwright's async API.

        Args:
            url (str): The URL to scrape.

        Returns:
            str: The scraped HTML content or an error message if an exception occurs.

        """
        from playwright.async_api import async_playwright

        logger.info("Starting scraping...")
        results = ""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url)
                results = await page.content()  # Simply get the HTML content
                logger.info("Content scraped")
            except Exception as e:
                results = f"Error: {e}"
            await browser.close()
        return results

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load text content from the provided URLs.

        This method yields Documents one at a time as they're scraped,
        instead of waiting to scrape all URLs before returning.

        Yields:
            Document: The scraped content encapsulated within a Document object.

        """
        for url in self.urls:
            html_content = asyncio.run(self.ascrape_playwright(url))
            metadata = {"source": url}
            yield Document(page_content=html_content, metadata=metadata)
