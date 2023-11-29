import itertools
import re
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from langchain_core.documents import Document

from langchain.document_loaders.web_base import WebBaseLoader


def _default_parsing_function(content: Any) -> str:
    return str(content.get_text())


def _default_meta_function(meta: dict, _content: Any) -> dict:
    return {"source": meta["loc"], **meta}


def _batch_block(iterable: Iterable, size: int) -> Generator[List[dict], None, None]:
    it = iter(iterable)
    while item := list(itertools.islice(it, size)):
        yield item


def _extract_scheme_and_domain(url: str) -> Tuple[str, str]:
    """Extract the scheme + domain from a given URL.

    Args:
        url (str): The input URL.

    Returns:
        return a 2-tuple of scheme and domain
    """
    parsed_uri = urlparse(url)
    return parsed_uri.scheme, parsed_uri.netloc


class SitemapLoader(WebBaseLoader):
    """Load a sitemap and its URLs.

    **Security Note**: This loader can be used to load all URLs specified in a sitemap.
        If a malicious actor gets access to the sitemap, they could force
        the server to load URLs from other domains by modifying the sitemap.
        This could lead to server-side request forgery (SSRF) attacks; e.g.,
        with the attacker forcing the server to load URLs from internal
        service endpoints that are not publicly accessible. While the attacker
        may not immediately gain access to this data, this data could leak
        into downstream systems (e.g., data loader is used to load data for indexing).

        This loader is a crawler and web crawlers should generally NOT be deployed
        with network access to any internal servers.

        Control access to who can submit crawling requests and what network access
        the crawler has.

        By default, the loader will only load URLs from the same domain as the sitemap
        if the site map is not a local file. This can be disabled by setting
        restrict_to_same_domain to False (not recommended).

        If the site map is a local file, no such risk mitigation is applied by default.

        Use the filter URLs argument to limit which URLs can be loaded.

        See https://python.langchain.com/docs/security
    """

    def __init__(
        self,
        web_path: str,
        filter_urls: Optional[List[str]] = None,
        parsing_function: Optional[Callable] = None,
        blocksize: Optional[int] = None,
        blocknum: int = 0,
        meta_function: Optional[Callable] = None,
        is_local: bool = False,
        continue_on_failure: bool = False,
        restrict_to_same_domain: bool = True,
        **kwargs: Any,
    ):
        """Initialize with webpage path and optional filter URLs.

        Args:
            web_path: url of the sitemap. can also be a local path
            filter_urls: a list of regexes. If specified, only
                URLS that match one of the filter URLs will be loaded.
                *WARNING* The filter URLs are interpreted as regular expressions.
                Remember to escape special characters if you do not want them to be
                interpreted as regular expression syntax. For example, `.` appears
                frequently in URLs and should be escaped if you want to match a literal
                `.` rather than any character.
                restrict_to_same_domain takes precedence over filter_urls when
                restrict_to_same_domain is True and the sitemap is not a local file.
            parsing_function: Function to parse bs4.Soup output
            blocksize: number of sitemap locations per block
            blocknum: the number of the block that should be loaded - zero indexed.
                Default: 0
            meta_function: Function to parse bs4.Soup output for metadata
                remember when setting this method to also copy metadata["loc"]
                to metadata["source"] if you are using this field
            is_local: whether the sitemap is a local file. Default: False
            continue_on_failure: whether to continue loading the sitemap if an error
                occurs loading a url, emitting a warning instead of raising an
                exception. Setting this to True makes the loader more robust, but also
                may result in missing data. Default: False
            restrict_to_same_domain: whether to restrict loading to URLs to the same
                domain as the sitemap. Attention: This is only applied if the sitemap
                is not a local file!
        """

        if blocksize is not None and blocksize < 1:
            raise ValueError("Sitemap blocksize should be at least 1")

        if blocknum < 0:
            raise ValueError("Sitemap blocknum can not be lower then 0")

        try:
            import lxml  # noqa:F401
        except ImportError:
            raise ImportError(
                "lxml package not found, please install it with `pip install lxml`"
            )

        super().__init__(web_paths=[web_path], **kwargs)

        # Define a list of URL patterns (interpreted as regular expressions) that
        # will be allowed to be loaded.
        # restrict_to_same_domain takes precedence over filter_urls when
        # restrict_to_same_domain is True and the sitemap is not a local file.
        self.allow_url_patterns = filter_urls
        self.restrict_to_same_domain = restrict_to_same_domain
        self.parsing_function = parsing_function or _default_parsing_function
        self.meta_function = meta_function or _default_meta_function
        self.blocksize = blocksize
        self.blocknum = blocknum
        self.is_local = is_local
        self.continue_on_failure = continue_on_failure

    def parse_sitemap(self, soup: Any) -> List[dict]:
        """Parse sitemap xml and load into a list of dicts.

        Args:
            soup: BeautifulSoup object.

        Returns:
            List of dicts.
        """
        els = []
        for url in soup.find_all("url"):
            loc = url.find("loc")
            if not loc:
                continue

            # Strip leading and trailing whitespace and newlines
            loc_text = loc.text.strip()

            if self.restrict_to_same_domain and not self.is_local:
                if _extract_scheme_and_domain(loc_text) != _extract_scheme_and_domain(
                    self.web_path
                ):
                    continue

            if self.allow_url_patterns and not any(
                re.match(regexp_pattern, loc_text)
                for regexp_pattern in self.allow_url_patterns
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
                raise ImportError(
                    "beautifulsoup4 package not found, please install it"
                    " with `pip install beautifulsoup4`"
                )
            fp = open(self.web_path)
            soup = bs4.BeautifulSoup(fp, "xml")
        else:
            soup = self._scrape(self.web_path, parser="xml")

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
