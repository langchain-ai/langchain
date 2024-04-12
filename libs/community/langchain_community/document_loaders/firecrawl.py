from typing import Iterator, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class FireCrawlLoader(BaseLoader):
    """Load from `Firecrawl.dev`."""

    def __init__(
        self,
        api_key: str,
        url: str,
        mode: Optional[str] = "crawl",
        params: Optional[dict] = None,
    ):
        """Initialize with API key and url.

        Args:
            api_key: The Firecrawl API key.
            url: The url to be crawled.
            mode: The mode to run the loader in. Default is "crawl".
                 Options include "scrape" (single url) and 
                 "crawl" (all accessible sub pages).
            params: The parameters to pass to the Firecrawl API.
                    Examples include crawlerOptions.
                    For more details, visit: https://github.com/mendableai/firecrawl-py
            wait_until_done: If True, waits until the crawl is done, returns the docs.
                             If False, returns jobId. Default is True.
        """

        try:
            from firecrawl import FirecrawlApp  # noqa: F401
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )
        self.firecrawl = FirecrawlApp(api_key=api_key)
        self.url = url
        self.mode = mode
        self.params = params

    def lazy_load(self) -> Iterator[Document]:
        if self.mode == "scrape":
            firecrawl_docs = self.firecrawl.scrape_url(self.url, params=self.params)
            yield Document(
                page_content=firecrawl_docs.get("markdown", ""),
                metadata=firecrawl_docs.get("metadata", {}),
            )
        else:
            firecrawl_docs = self.firecrawl.crawl_url(self.url, params=self.params)
            for doc in firecrawl_docs:
                yield Document(
                    page_content=doc.get("markdown", ""),
                    metadata=doc.get("metadata", {}),
                )
