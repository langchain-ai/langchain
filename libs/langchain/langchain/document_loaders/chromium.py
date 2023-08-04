import asyncio
import logging
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
from typing import Iterator, List

logger = logging.getLogger(__name__)

class AsyncChromiumLoader(BaseLoader):
    """Scrape HTML using a Headless instance of Chromium."""

    def __init__(
        self,
        urls: List[str],
    ):
        """Initialize with URL paths."""
        self.urls = urls

    async def ascrape_playwright(self, url: str) -> str:
        logger.info("Starting scraping...")
        results = ""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url)
                results = await page.content() # Simply get the HTML content
                logger.info("Content scraped")
            except Exception as e:
                results = f"Error: {e}"
            await browser.close()
        return results

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in url."""
        for url in self.urls:
            html_content = asyncio.run(self.ascrape_playwright(url))
            metadata = {"source": url}
            yield Document(page_content=html_content, metadata=metadata)

    def load(self) -> List[Document]:
        """Load Documents from URLs."""
        return list(self.lazy_load())
