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
        """Initialize with urk paths."""
        self.urls = urls

    def remove_unwanted_tags(self, html_content, unwanted_tags=["script", "style"]):
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        return str(soup)

    def extract_tags(self, html_content, tags: list[str]):
        soup = BeautifulSoup(html_content, "html.parser")
        text_parts = []
        for tag in tags:
            elements = soup.find_all(tag)
            for element in elements:
                if tag == "a":
                    href = element.get("href")
                    if href:
                        text_parts.append(f"{element.get_text()} ({href})")
                    else:
                        text_parts.append(element.get_text())
                else:
                    text_parts.append(element.get_text())

        return " ".join(text_parts)

    def scrape_by_url_raw(url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return response.text

    def remove_unessesary_lines(self, content: str) -> str:
        lines = content.split("\n")
        stripped_lines = [line.strip() for line in lines]
        non_empty_lines = [line for line in stripped_lines if line]
        seen = set()
        deduped_lines = [
            line for line in non_empty_lines if not (line in seen or seen.add(line))
        ]
        cleaned_content = "".join(deduped_lines)

        return cleaned_content

    async def ascrape_playwright(self, url: str) -> str:
        logger.info("Starting scraping...")
        results = ""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url)
                page_source = await page.content()
                results = self.remove_unessesary_lines(
                    self.extract_tags(
                        self.remove_unwanted_tags(page_source), ["p", "li", "div", "a"]
                    )
                )
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
