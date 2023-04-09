"""Loader that fetches a sitemap and loads those URLs."""
import re
from typing import Any, Callable, List, Optional

from langchain.document_loaders.web_base import WebBaseLoader
from langchain.schema import Document


def _default_parsing_function(content: Any) -> str:
    return str(content.get_text())


class SitemapLoader(WebBaseLoader):
    """Loader that fetches a sitemap and loads those URLs."""

    def __init__(
        self,
        web_path: str,
        filter_urls: Optional[List[str]] = None,
        parsing_function: Optional[Callable] = None,
    ):
        """Initialize with webpage path and optional filter URLs.

        Args:
            web_path: url of the sitemap
            filter_urls: list of strings or regexes that will be applied to filter the
                urls that are parsed and loaded
            parsing_function: Function to parse bs4.Soup output
        """

        try:
            import lxml  # noqa:F401
        except ImportError:
            raise ValueError(
                "lxml package not found, please install it with " "`pip install lxml`"
            )

        super().__init__(web_path)

        self.filter_urls = filter_urls
        self.parsing_function = parsing_function or _default_parsing_function

    def parse_sitemap(self, soup: Any) -> List[dict]:
        """Parse sitemap xml and load into a list of dicts."""
        els = []
        for url in soup.find_all("url"):
            loc = url.find("loc")
            if not loc:
                continue

            if self.filter_urls and not any(
                re.match(r, loc.text) for r in self.filter_urls
            ):
                continue

            els.append(
                {
                    tag: prop.text
                    for tag in ["loc", "lastmod", "changefreq", "priority"]
                    if (prop := url.find(tag))
                }
            )

        return els

    def load(self) -> List[Document]:
        """Load sitemap."""
        soup = self.scrape("xml")

        els = self.parse_sitemap(soup)

        results = self.scrape_all([el["loc"].strip() for el in els if "loc" in el])

        return [
            Document(
                page_content=self.parsing_function(results[i]),
                metadata={**{"source": els[i]["loc"]}, **els[i]},
            )
            for i in range(len(results))
        ]
