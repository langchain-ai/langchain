"""ScrapingAnt Web Extractor."""

import logging
from typing import Iterator, List, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_env

logger = logging.getLogger(__file__)


class ScrapingAntLoader(BaseLoader):
    """Turn an url to LLM accessible markdown with `ScrapingAnt`.

    For further details, visit: https://docs.scrapingant.com/python-client
    """

    def __init__(
        self,
        urls: List[str],
        *,
        api_key: Optional[str] = None,
        scrape_config: Optional[dict] = None,
        continue_on_failure: bool = True,
    ) -> None:
        """Initialize client.

        Args:
            urls: List of urls to scrape.
            api_key: The ScrapingAnt API key. If not specified must have env var
                SCRAPINGANT_API_KEY set.
            scrape_config: The scraping config from ScrapingAntClient.markdown_request
            continue_on_failure: Whether to continue if scraping an url fails.
        """
        try:
            from scrapingant_client import ScrapingAntClient
        except ImportError:
            raise ImportError(
                "`scrapingant-client` package not found,"
                " run `pip install scrapingant-client`"
            )
        if not urls:
            raise ValueError("URLs must be provided.")
        api_key = api_key or get_from_env("api_key", "SCRAPINGANT_API_KEY")
        self.client = ScrapingAntClient(token=api_key)
        self.urls = urls
        self.scrape_config = scrape_config
        self.continue_on_failure = continue_on_failure

    def lazy_load(self) -> Iterator[Document]:
        """Fetch data from ScrapingAnt."""

        scrape_config = self.scrape_config if self.scrape_config is not None else {}
        for url in self.urls:
            try:
                result = self.client.markdown_request(url=url, **scrape_config)
                yield Document(
                    page_content=result.markdown,
                    metadata={"url": result.url},
                )
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching data from {url}, exception: {e}")
                else:
                    raise e
