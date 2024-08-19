from typing import Iterator, Literal, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_env


class FireCrawlLoader(BaseLoader):
    """
    FireCrawlLoader document loader integration

    Setup:
        Install ``firecrawl-py``,``langchain_community`` and set environment variable ``FIRECRAWL_API_KEY``.

        .. code-block:: bash

            pip install -U firecrawl-py langchain_community
            export FIRECRAWL_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import FireCrawlLoader

            loader = FireCrawlLoader(
                url = "https://firecrawl.dev",
                mode = "crawl"
                # other params = ...
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Introducing [Smart Crawl!](https://www.firecrawl.dev/smart-crawl)
             Join the waitlist to turn any web
            {'ogUrl': 'https://www.firecrawl.dev/', 'title': 'Home - Firecrawl', 'robots': 'follow, index', 'ogImage': 'https://www.firecrawl.dev/og.png?123', 'ogTitle': 'Firecrawl', 'sitemap': {'lastmod': '2024-08-12T00:28:16.681Z', 'changefreq': 'weekly'}, 'keywords': 'Firecrawl,Markdown,Data,Mendable,Langchain', 'sourceURL': 'https://www.firecrawl.dev/', 'ogSiteName': 'Firecrawl', 'description': 'Firecrawl crawls and converts any website into clean markdown.', 'ogDescription': 'Turn any website into LLM-ready data.', 'pageStatusCode': 200, 'ogLocaleAlternate': []}


    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Introducing [Smart Crawl!](https://www.firecrawl.dev/smart-crawl)
             Join the waitlist to turn any web
            {'ogUrl': 'https://www.firecrawl.dev/', 'title': 'Home - Firecrawl', 'robots': 'follow, index', 'ogImage': 'https://www.firecrawl.dev/og.png?123', 'ogTitle': 'Firecrawl', 'sitemap': {'lastmod': '2024-08-12T00:28:16.681Z', 'changefreq': 'weekly'}, 'keywords': 'Firecrawl,Markdown,Data,Mendable,Langchain', 'sourceURL': 'https://www.firecrawl.dev/', 'ogSiteName': 'Firecrawl', 'description': 'Firecrawl crawls and converts any website into clean markdown.', 'ogDescription': 'Turn any website into LLM-ready data.', 'pageStatusCode': 200, 'ogLocaleAlternate': []}

    """  # noqa: E501

    def __init__(
        self,
        url: str,
        *,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        mode: Literal["crawl", "scrape"] = "crawl",
        params: Optional[dict] = None,
    ):
        """Initialize with API key and url.

        Args:
            url: The url to be crawled.
            api_key: The Firecrawl API key. If not specified will be read from env var
                FIRECRAWL_API_KEY. Get an API key
            api_url: The Firecrawl API URL. If not specified will be read from env var
                FIRECRAWL_API_URL or defaults to https://api.firecrawl.dev.
            mode: The mode to run the loader in. Default is "crawl".
                 Options include "scrape" (single url) and
                 "crawl" (all accessible sub pages).
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
        if mode not in ("crawl", "scrape"):
            raise ValueError(
                f"Unrecognized mode '{mode}'. Expected one of 'crawl', 'scrape'."
            )
        api_key = api_key or get_from_env("api_key", "FIRECRAWL_API_KEY")
        self.firecrawl = FirecrawlApp(api_key=api_key, api_url=api_url)
        self.url = url
        self.mode = mode
        self.params = params

    def lazy_load(self) -> Iterator[Document]:
        if self.mode == "scrape":
            firecrawl_docs = [self.firecrawl.scrape_url(self.url, params=self.params)]
        elif self.mode == "crawl":
            firecrawl_docs = self.firecrawl.crawl_url(self.url, params=self.params)
        else:
            raise ValueError(
                f"Unrecognized mode '{self.mode}'. Expected one of 'crawl', 'scrape'."
            )
        for doc in firecrawl_docs:
            metadata = doc.get("metadata", {})
            if (self.params is not None) and self.params.get(
                "extractorOptions", {}
            ).get("mode") == "llm-extraction":
                metadata["llm_extraction"] = doc.get("llm_extraction")

            yield Document(page_content=doc.get("markdown", ""), metadata=metadata)
