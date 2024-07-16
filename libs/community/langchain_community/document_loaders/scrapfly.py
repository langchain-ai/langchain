"""Scrapfly Web Reader."""

import logging
from typing import Iterator, List, Literal, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_env

logger = logging.getLogger(__file__)


class ScrapflyLoader(BaseLoader):
    """Turn a url to llm accessible markdown with `Scrapfly.io`.

    For further details, visit: https://scrapfly.io/docs/sdk/python
    """

    def __init__(
        self,
        urls: List[str],
        *,
        api_key: Optional[str] = None,
        scrape_format: Literal["markdown", "text"] = "markdown",
        scrape_config: Optional[dict] = None,
        continue_on_failure: bool = True,
    ) -> None:
        """Initialize client.

        Args:
            urls: List of urls to scrape.
            api_key: The Scrapfly API key. If not specified must have env var
                SCRAPFLY_API_KEY set.
            scrape_format: Scrape result format, one or "markdown" or "text".
            scrape_config: Dictionary of ScrapFly scrape config object.
            continue_on_failure: Whether to continue if scraping a url fails.
        """
        try:
            from scrapfly import ScrapflyClient
        except ImportError:
            raise ImportError(
                "`scrapfly` package not found, please run `pip install scrapfly-sdk`"
            )
        if not urls:
            raise ValueError("URLs must be provided.")
        api_key = api_key or get_from_env("api_key", "SCRAPFLY_API_KEY")
        self.scrapfly = ScrapflyClient(key=api_key)
        self.urls = urls
        self.scrape_format = scrape_format
        self.scrape_config = scrape_config
        self.continue_on_failure = continue_on_failure

    def lazy_load(self) -> Iterator[Document]:
        from scrapfly import ScrapeConfig

        scrape_config = self.scrape_config if self.scrape_config is not None else {}
        for url in self.urls:
            try:
                response = self.scrapfly.scrape(
                    ScrapeConfig(url, format=self.scrape_format, **scrape_config)
                )
                yield Document(
                    page_content=response.scrape_result["content"],
                    metadata={"url": url},
                )
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching data from {url}, exception: {e}")
                else:
                    raise e
