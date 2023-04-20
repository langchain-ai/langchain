"""Loader that fetches a sitemap and loads those URLs."""
import re
from typing import Any, Callable, List, Optional
from urllib.parse import urlparse

from langchain.document_loaders.web_base import WebBaseLoader
from langchain.schema import Document


def _default_parsing_function(content: Any) -> str:
    return str(content.get_text())


class SitemapLoader(WebBaseLoader):
    """Loader that fetches a sitemap and loads those URLs."""

    def __init__(
        self,
        web_path: str,
        discover_sitemap: bool = False,
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
        if discover_sitemap:
            self.web_paths = self._modify_web_path()

    # if web_path is a str
    @property
    def _base_url(self) -> str:
        base_url = (
            f"{urlparse(self.web_path).scheme}"
            + "://"
            + f"{urlparse(self.web_path).netloc}"
        )
        return base_url

    def _find_sitemap_in_robotstxt(self) -> List[str]:
        import requests

        """Find sitemap in robots.txt."""
        sitemap_urls: List[str] = []
        robots_txt_url = self._base_url + "/robots.txt"
        response = requests.get(robots_txt_url)
        if response.status_code == 200:
            site_map_urls = re.findall(r"Sitemap: (\S+)", response.text, re.IGNORECASE)
        return site_map_urls

    def _find_sitemap_in_html(self) -> List[str]:
        """Find sitemap in homepage html."""
        import requests
        from bs4 import BeautifulSoup

        sitemap_urls = []

        response = requests.get(self._base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a"):
                if "sitemap" in link.get("href").lower():
                    sitemap_urls.append(self._base_url + link.get("href"))
        return sitemap_urls

    def _modify_web_path(self) -> List[str]:
        sitemap_urls = self._find_sitemap_in_robotstxt()
        if not sitemap_urls:
            sitemap_urls = self._find_sitemap_in_html()
        if not sitemap_urls:
            raise ValueError("No sitemap found in robots.txt or the homepage html.")
        return sitemap_urls

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
