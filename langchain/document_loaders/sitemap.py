"""Loader that fetches a sitemap and loads those URLs."""
import itertools
import re
from typing import Any, Callable, Generator, Iterable, List, Optional

from langchain.document_loaders.web_base import WebBaseLoader
from langchain.schema import Document


def _default_parsing_function(content: Any) -> str:
    return str(content.get_text())


def _default_meta_function(meta: dict, _content: Any) -> dict:
    return {"source": meta["loc"], **meta}


def _batch_block(iterable: Iterable, size: int) -> Generator[List[dict], None, None]:
    it = iter(iterable)
    while item := list(itertools.islice(it, size)):
        yield item


class SitemapLoader(WebBaseLoader):
    """Loader that fetches a sitemap and loads those URLs."""

    def __init__(
        self,
        web_path: str,
        filter_urls: Optional[List[str]] = None,
        parsing_function: Optional[Callable] = None,
        blocksize: Optional[int] = None,
        blocknum: int = 0,
        meta_function: Optional[Callable] = None,
        is_local: bool = False,
    ):
        """Initialize with webpage path and optional filter URLs.

        Args:
            web_path: url of the sitemap. can also be a local path
            filter_urls: list of strings or regexes that will be applied to filter the
                urls that are parsed and loaded
            parsing_function: Function to parse bs4.Soup output
            blocksize: number of sitemap locations per block
            blocknum: the number of the block that should be loaded - zero indexed
            meta_function: Function to parse bs4.Soup output for metadata
                remember when setting this method to also copy metadata["loc"]
                to metadata["source"] if you are using this field
            is_local: whether the sitemap is a local file
        """

        if blocksize is not None and blocksize < 1:
            raise ValueError("Sitemap blocksize should be at least 1")

        if blocknum < 0:
            raise ValueError("Sitemap blocknum can not be lower then 0")

        try:
            import lxml  # noqa:F401
        except ImportError:
            raise ValueError(
                "lxml package not found, please install it with " "`pip install lxml`"
            )

        super().__init__(web_path)

        self.filter_urls = filter_urls
        self.parsing_function = parsing_function or _default_parsing_function
        self.meta_function = meta_function or _default_meta_function
        self.blocksize = blocksize
        self.blocknum = blocknum
        self.is_local = is_local

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

        for sitemap in soup.find_all("sitemap"):
            loc = sitemap.find("loc")
            if not loc:
                continue
            soup_child = self.scrape_all([loc.text], "xml")[0]

            els.extend(self.parse_sitemap(soup_child))
        return els

    def load(self) -> List[Document]:
        """Load sitemap."""
        if self.is_local:
            try:
                import bs4
            except ImportError:
                raise ValueError(
                    "bs4 package not found, please install it with " "`pip install bs4`"
                )
            fp = open(self.web_path)
            soup = bs4.BeautifulSoup(fp, "xml")
        else:
            soup = self.scrape("xml")

        els = self.parse_sitemap(soup)

        if self.blocksize is not None:
            elblocks = list(_batch_block(els, self.blocksize))
            blockcount = len(elblocks)
            if blockcount - 1 < self.blocknum:
                raise ValueError(
                    "Selected sitemap does not contain enough blocks for given blocknum"
                )
            else:
                els = elblocks[self.blocknum]

        results = self.scrape_all([el["loc"].strip() for el in els if "loc" in el])

        return [
            Document(
                page_content=self.parsing_function(results[i]),
                metadata=self.meta_function(els[i], results[i]),
            )
            for i in range(len(results))
        ]
