from typing import Iterator, Literal, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_env


class FireCrawlLoader(BaseLoader):
    """Load web pages as Documents using FireCrawl.

    Must have Python package `firecrawl` installed and a FireCrawl API key. See
        https://www.firecrawl.dev/ for more.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        mode: Literal["crawl", "scrape", "search"] = "crawl",
        params: Optional[dict] = None,
    ):
        """Initialize with API key and url.

        Args:
            url: The url to be crawled.
            query: The query to be used for the search.
            api_key: The Firecrawl API key. If not specified will be read from env var
                FIREWALL_API_KEY. Get an API key
            mode: The mode to run the loader in. Default is "crawl".
                 Options include "scrape" (single url), "crawl" (all accessible sub pages),
                 and "search" (search for the query).
            params: The parameters to pass to the Firecrawl API.
                Examples include crawlerOptions.
                For more details, visit: https://github.com/mendableai/firecrawl-py
        """

        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )
        if mode not in ("crawl", "scrape", "search"):
            raise ValueError(
                f"Unrecognized mode '{mode}'. Expected one of 'crawl', 'scrape', 'search'."
            )

        if not query and not url:
            raise ValueError("Either query or url must be provided")
        elif query and url:
            raise ValueError("Only one of query or url can be provided")

        api_key = api_key or get_from_env("api_key", "FIREWALL_API_KEY")
        self.firecrawl = FirecrawlApp(api_key=api_key)
        self.url = url
        self.query = query
        self.mode = mode
        self.params = params

    def lazy_load(self) -> Iterator[Document]:
        if self.mode == "scrape":
            if not self.url:
                raise ValueError("URL is required for scrape mode")
            firecrawl_docs = [self.firecrawl.scrape_url(self.url, params=self.params)]
        elif self.mode == "crawl":
            if not self.url:
                raise ValueError("URL is required for crawl mode")
            firecrawl_docs = self.firecrawl.crawl_url(self.url, params=self.params)
        elif self.mode == "search":
            if not self.query:
                raise ValueError("Query is required for search mode")
            firecrawl_docs = self.firecrawl.search(self.query, params=self.params)
        else:
            raise ValueError(
                f"Unrecognized mode '{self.mode}'. Expected one of 'crawl', 'scrape' or 'search'."
            )
        for doc in firecrawl_docs:
            yield Document(
                page_content=doc.get("markdown", ""),
                metadata=doc.get("metadata", {}),
            )
