import warnings
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

    def legacy_crawler_options_adapter(self, params: dict) -> dict:
        use_legacy_options = False
        legacy_keys = [
            "includes",
            "excludes",
            "allowBackwardCrawling",
            "allowExternalContentLinks",
            "pageOptions",
        ]
        for key in legacy_keys:
            if params.get(key):
                use_legacy_options = True
                break

        if use_legacy_options:
            warnings.warn(
                "Deprecated parameters detected. See Firecrawl v1 docs for updates.",
                DeprecationWarning,
            )
            if "includes" in params:
                if params["includes"] is True:
                    params["includePaths"] = params["includes"]
                del params["includes"]

            if "excludes" in params:
                if params["excludes"] is True:
                    params["excludePaths"] = params["excludes"]
                del params["excludes"]

            if "allowBackwardCrawling" in params:
                if params["allowBackwardCrawling"] is True:
                    params["allowBackwardLinks"] = params["allowBackwardCrawling"]
                del params["allowBackwardCrawling"]

            if "allowExternalContentLinks" in params:
                if params["allowExternalContentLinks"] is True:
                    params["allowExternalLinks"] = params["allowExternalContentLinks"]
                del params["allowExternalContentLinks"]

            if "pageOptions" in params:
                if isinstance(params["pageOptions"], dict):
                    params["scrapeOptions"] = self.legacy_scrape_options_adapter(
                        params["pageOptions"]
                    )
                del params["pageOptions"]

        return params

    def legacy_scrape_options_adapter(self, params: dict) -> dict:
        use_legacy_options = False
        formats = ["markdown"]

        if "extractorOptions" in params:
            if "mode" in params["extractorOptions"]:
                if (
                    params["extractorOptions"]["mode"] == "llm-extraction"
                    or params["extractorOptions"]["mode"]
                    == "llm-extraction-from-raw-html"
                    or params["extractorOptions"]["mode"]
                    == "llm-extraction-from-markdown"
                ):
                    use_legacy_options = True
                    if "extractionPrompt" in params["extractorOptions"]:
                        if params["extractorOptions"]["extractionPrompt"]:
                            params["prompt"] = params["extractorOptions"][
                                "extractionPrompt"
                            ]
                        else:
                            params["prompt"] = params["extractorOptions"].get(
                                "extractionPrompt",
                                "Extract page information based on the schema.",
                            )

                    if "extractionSchema" in params["extractorOptions"]:
                        if params["extractorOptions"]["extractionSchema"]:
                            params["schema"] = params["extractorOptions"][
                                "extractionSchema"
                            ]

                    if "userPrompt" in params["extractorOptions"]:
                        if params["extractorOptions"]["userPrompt"]:
                            params["prompt"] = params["extractorOptions"]["userPrompt"]

                    del params["extractorOptions"]

        scrape_keys = [
            "includeMarkdown",
            "includeHtml",
            "includeRawHtml",
            "includeExtract",
            "includeLinks",
            "screenshot",
            "fullPageScreenshot",
            "onlyIncludeTags",
            "removeTags",
        ]
        for key in scrape_keys:
            if params.get(key):
                use_legacy_options = True
                break

        if use_legacy_options:
            warnings.warn(
                "Deprecated parameters detected. See Firecrawl v1 docs for updates.",
                DeprecationWarning,
            )
            if "includeMarkdown" in params:
                if params["includeMarkdown"] is False:
                    formats.remove("markdown")
                del params["includeMarkdown"]

            if "includeHtml" in params:
                if params["includeHtml"] is True:
                    formats.append("html")
                del params["includeHtml"]

            if "includeRawHtml" in params:
                if params["includeRawHtml"] is True:
                    formats.append("rawHtml")
                del params["includeRawHtml"]

            if "includeExtract" in params:
                if params["includeExtract"] is True:
                    formats.append("extract")
                del params["includeExtract"]

            if "includeLinks" in params:
                if params["includeLinks"] is True:
                    formats.append("links")
                del params["includeLinks"]

            if "screenshot" in params:
                if params["screenshot"] is True:
                    formats.append("screenshot")
                del params["screenshot"]

            if "fullPageScreenshot" in params:
                if params["fullPageScreenshot"] is True:
                    formats.append("screenshot@fullPage")
                del params["fullPageScreenshot"]

            if "onlyIncludeTags" in params:
                if params["onlyIncludeTags"] is True:
                    params["includeTags"] = params["onlyIncludeTags"]
                del params["onlyIncludeTags"]

            if "removeTags" in params:
                if params["removeTags"] is True:
                    params["excludeTags"] = params["removeTags"]
                del params["removeTags"]

        if "formats" not in params:
            params["formats"] = formats

        return params

    def __init__(
        self,
        url: str,
        *,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        mode: Literal["crawl", "scrape", "map"] = "crawl",
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
                 Options include "scrape" (single url),
                 "crawl" (all accessible sub pages),
                 "map" (returns list of links that are semantically related).
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
        if mode not in ("crawl", "scrape", "search", "map"):
            raise ValueError(
                f"Invalid mode '{mode}'. Allowed: 'crawl', 'scrape', 'search', 'map'."
            )

        if not url:
            raise ValueError("Url must be provided")

        api_key = api_key or get_from_env("api_key", "FIRECRAWL_API_KEY")
        self.firecrawl = FirecrawlApp(api_key=api_key, api_url=api_url)
        self.url = url
        self.mode = mode
        self.params = params or {}

    def lazy_load(self) -> Iterator[Document]:
        if self.mode == "scrape":
            firecrawl_docs = [
                self.firecrawl.scrape_url(
                    self.url, params=self.legacy_scrape_options_adapter(self.params)
                )
            ]
        elif self.mode == "crawl":
            if not self.url:
                raise ValueError("URL is required for crawl mode")
            crawl_response = self.firecrawl.crawl_url(
                self.url, params=self.legacy_crawler_options_adapter(self.params)
            )
            firecrawl_docs = crawl_response.get("data", [])
        elif self.mode == "map":
            if not self.url:
                raise ValueError("URL is required for map mode")
            firecrawl_docs = self.firecrawl.map_url(self.url, params=self.params)
        elif self.mode == "search":
            raise ValueError(
                "Search mode is not supported in this version, please downgrade."
            )
        else:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Allowed: 'crawl', 'scrape', 'map'."
            )
        for doc in firecrawl_docs:
            if self.mode == "map":
                page_content = doc
                metadata = {}
            else:
                page_content = (
                    doc.get("markdown") or doc.get("html") or doc.get("rawHtml", "")
                )
                metadata = doc.get("metadata", {})
            if not page_content:
                continue
            yield Document(
                page_content=page_content,
                metadata=metadata,
            )
