"""Scrapfly Web Reader."""
import logging
from typing import Iterator, List, Optional, Literal

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_env

logger = logging.getLogger(__file__)


class ScrapflyLoader(BaseLoader):
    """Turn a url to llm accessible markdown with `Scrapfly.io`.

    Args:
    api_key: The Scrapfly API key.
    urls: List of urls to scrape.
    scrape_format: Scrape result format (markdown or text)
    scrape_config: Optional[dict]: Dictionary of ScrapFly scrape config object.
    ignore_scrape_failures: Whether to continue on failures.
    For further details, visit: https://scrapfly.io/docs/sdk/python
    """

    api_key: str
    urls: List[str]
    scrape_format: Literal["markdown", "text"]
    scrape_config: Optional[dict]
    ignore_scrape_failures: bool

    def __init__(
        self,
        api_key: str,
        urls: List[str],
        scrape_format: Literal["markdown", "text"] = "markdown",
        scrape_config: Optional[dict] = None,
        ignore_scrape_failures: bool = True,
    ) -> None:
        """Initialize client."""
        try:
            from scrapfly import ScrapflyClient
        except ImportError:
            raise ImportError(
                "`scrapfly` package not found, please run `pip install scrapfly-sdk`"
            )
        api_key = api_key or get_from_env("api_key", "SCRAPFLY_API_KEY")
        self.scrapfly = ScrapflyClient(key=api_key)
        self.urls = urls
        self.scrape_format = scrape_format
        self.scrape_config = scrape_config
        self.ignore_scrape_failures = ignore_scrape_failures

    @classmethod
    def class_name(cls) -> str:
        return "Scrapfly_reader"

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        from scrapfly import ScrapeApiResponse, ScrapeConfig

        if self.urls is None:
            raise ValueError("URLs must be provided.")
        scrape_config = self.scrape_config if self.scrape_config is not None else {}
        documents = []

        try:
            for url in self.urls:
                response: ScrapeApiResponse = self.scrapfly.scrape(
                    ScrapeConfig(url, format=self.scrape_format, **scrape_config)
                )
                documents.append(
                    Document(
                        page_content=response.scrape_result["content"],
                        extra_info={"url": url},
                    )
                )
        except Exception as e:
            if self.ignore_scrape_failures:
                logger.error(f"Error fetching data from {url}, exception: {e}")
            else:
                raise e

        for doc in documents:
            yield doc
