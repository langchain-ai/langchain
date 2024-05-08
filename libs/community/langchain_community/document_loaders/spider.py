from typing import Iterator, Literal, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_env


class SpiderLoader(BaseLoader):
    """Load web pages as Documents using Spider AI.

    Must have the Python package `spider-client` installed and a Spider API key.
    See https://spider.cloud for more.
    """

    def __init__(
        self,
        url: str,
        *,
        api_key: Optional[str] = None,
        mode: Literal["scrape", "crawl"] = "scrape",
        params: Optional[dict] = {"return_format": "markdown"},
    ):
        """Initialize with API key and URL.

        Args:
            url: The URL to be processed.
            api_key: The Spider API key. If not specified, will be read from env
            var `SPIDER_API_KEY`.
            mode: The mode to run the loader in. Default is "scrape".
                 Options include "scrape" (single page) and "crawl" (with deeper
                 crawling following subpages).
            params: Additional parameters for the Spider API.
        """
        try:
            from spider import Spider
        except ImportError:
            raise ImportError(
                "`spider` package not found, please run `pip install spider-client`"
            )
        if mode not in ("scrape", "crawl"):
            raise ValueError(
                f"Unrecognized mode '{mode}'. Expected one of 'scrape', 'crawl'."
            )
        # If `params` is `None`, initialize it as an empty dictionary
        if params is None:
            params = {}

        # Add a default value for 'metadata' if it's not already present
        if "metadata" not in params:
            params["metadata"] = True

        # Use the environment variable if the API key isn't provided
        api_key = api_key or get_from_env("api_key", "SPIDER_API_KEY")
        self.spider = Spider(api_key=api_key)
        self.url = url
        self.mode = mode
        self.params = params

    def lazy_load(self) -> Iterator[Document]:
        """Load documents based on the specified mode."""
        spider_docs = []

        if self.mode == "scrape":
            # Scrape a single page
            response = self.spider.scrape_url(self.url, params=self.params)
            if response:
                spider_docs.append(response)
        elif self.mode == "crawl":
            # Crawl multiple pages
            response = self.spider.crawl_url(self.url, params=self.params)
            if response:
                spider_docs.extend(response)

        for doc in spider_docs:
            if self.mode == "scrape":
                # Ensure page_content is also not None
                page_content = doc[0].get("content", "")

                # Ensure metadata is also not None
                metadata = doc[0].get("metadata", {})

                yield Document(page_content=page_content, metadata=metadata)
            if self.mode == "crawl":
                # Ensure page_content is also not None
                page_content = doc.get("content", "")

                # Ensure metadata is also not None
                metadata = doc.get("metadata", {})

                if page_content is not None:
                    yield Document(
                        page_content=page_content,
                        metadata=metadata,
                    )
