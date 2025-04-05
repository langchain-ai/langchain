import warnings
from typing import Any, AsyncIterator, Iterator, List, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from langchain_community.document_loaders.web_base import WebBaseLoader


class GitbookLoader(WebBaseLoader):
    """Load `GitBook` data.

    1. load from either a single page, or
    2. load all (relative) paths in the sitemap, handling nested sitemap indexes.
    """

    def __init__(
        self,
        web_page: str,
        load_all_paths: bool = False,
        base_url: Optional[str] = None,
        content_selector: str = "main",
        continue_on_failure: bool = False,
        show_progress: bool = True,
        *,
        sitemap_url: Optional[str] = None,
    ):
        """Initialize with web page and whether to load all paths.

        Args:
            web_page: The web page to load or the starting point from where
                relative paths are discovered.
            load_all_paths: If set to True, all relative paths in the navbar
                are loaded instead of only `web_page`.
            base_url: If `load_all_paths` is True, the relative paths are
                appended to this base url. Defaults to `web_page`.
            content_selector: The CSS selector for the content to load.
                Defaults to "main".
            continue_on_failure: whether to continue loading the sitemap if an error
                occurs loading a url, emitting a warning instead of raising an
                exception. Setting this to True makes the loader more robust, but also
                may result in missing data. Default: False
            show_progress: whether to show a progress bar while loading. Default: True
            sitemap_url: Custom sitemap URL to use when load_all_paths is True.
                Defaults to "{base_url}/sitemap.xml".
        """
        self.base_url = base_url or web_page
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

        if load_all_paths:
            # set web_path to the sitemap if we want to crawl all paths
            if sitemap_url:
                web_page = sitemap_url
            else:
                web_page = f"{self.base_url}/sitemap.xml"

        super().__init__(
            web_paths=(web_page,),
            continue_on_failure=continue_on_failure,
            show_progress=show_progress,
        )
        self.load_all_paths = load_all_paths
        self.content_selector = content_selector

    def _is_sitemap_index(self, soup: BeautifulSoup) -> bool:
        """Check if the soup contains a sitemap index."""
        return soup.find("sitemapindex") is not None

    def _extract_sitemap_urls(self, soup: BeautifulSoup) -> List[str]:
        """Extract sitemap URLs from a sitemap index."""
        sitemap_tags = soup.find_all("sitemap")
        urls = []
        for sitemap in sitemap_tags:
            loc = sitemap.find("loc")
            if loc and loc.text:
                urls.append(loc.text)
        return urls

    def _process_sitemap(
        self, soup: BeautifulSoup, processed_urls: Optional[Set[str]] = None
    ) -> List[str]:
        """Process a sitemap, handling both direct content URLs and sitemap indexes."""
        if processed_urls is None:
            processed_urls = set()

        # If it's a sitemap index, recursively process each sitemap URL
        if self._is_sitemap_index(soup):
            sitemap_urls = self._extract_sitemap_urls(soup)
            all_content_urls = []

            for sitemap_url in sitemap_urls:
                if sitemap_url in processed_urls:
                    warnings.warn(
                        f"Skipping already processed sitemap URL: {sitemap_url}"
                    )
                    continue

                processed_urls.add(sitemap_url)
                try:
                    # We need to temporarily set the web_paths to the sitemap URL
                    original_web_paths = self.web_paths
                    self.web_paths = [sitemap_url]
                    sitemap_soup = self.scrape(parser="xml")
                    # Restore original web_paths
                    self.web_paths = original_web_paths
                    content_urls = self._process_sitemap(sitemap_soup, processed_urls)
                    all_content_urls.extend(content_urls)
                except Exception as e:
                    if self.continue_on_failure:
                        warnings.warn(f"Error processing sitemap {sitemap_url}: {e}")
                    else:
                        raise

            return all_content_urls
        else:
            # It's a content sitemap, so extract content URLs
            return self._get_paths(soup)

    async def _aprocess_sitemap(
        self,
        soup: BeautifulSoup,
        base_url: str,
        processed_urls: Optional[Set[str]] = None,
    ) -> List[str]:
        """Async version of _process_sitemap."""
        if processed_urls is None:
            processed_urls = set()

        # If it's a sitemap index, recursively process each sitemap URL
        if self._is_sitemap_index(soup):
            sitemap_urls = self._extract_sitemap_urls(soup)
            all_content_urls = []

            # Use base class's ascrape_all for efficient parallel fetching
            soups = await self.ascrape_all(
                [url for url in sitemap_urls if url not in processed_urls], parser="xml"
            )
            for sitemap_url, sitemap_soup in zip(
                [url for url in sitemap_urls if url not in processed_urls], soups
            ):
                processed_urls.add(sitemap_url)
                try:
                    content_urls = await self._aprocess_sitemap(
                        sitemap_soup, base_url, processed_urls
                    )
                    all_content_urls.extend(content_urls)
                except Exception as e:
                    if self.continue_on_failure:
                        warnings.warn(f"Error processing sitemap {sitemap_url}: {e}")
                    else:
                        raise

            return all_content_urls
        else:
            # It's a content sitemap, so extract content URLs
            return self._get_paths(soup)

    def lazy_load(self) -> Iterator[Document]:
        """Fetch text from one single GitBook page or recursively from sitemap."""
        if self.load_all_paths:
            # Get initial sitemap
            soup_info = self.scrape()

            # Process sitemap(s) recursively to get all content URLs
            relative_paths = self._process_sitemap(soup_info)
            if not relative_paths and self.show_progress:
                warnings.warn(
                    f"No content URLs found in sitemap at {self.web_paths[0]}"
                )

            urls = [urljoin(self.base_url, path) for path in relative_paths]

            # Use base class's scrape_all to efficiently fetch all pages
            soup_infos = self.scrape_all(urls)

            for soup_info, url in zip(soup_infos, urls):
                doc = self._get_document(soup_info, url)
                if doc:
                    yield doc
        else:
            # Use base class functionality directly for single page
            for doc in super().lazy_load():
                yield doc

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Asynchronously fetch text from GitBook page(s)."""
        if not self.load_all_paths:
            # For single page case, use the parent class implementation
            async for doc in super().alazy_load():
                yield doc
        else:
            # Fetch initial sitemap using base class's functionality
            soups = await self.ascrape_all(self.web_paths, parser="xml")
            soup_info = soups[0]

            # Process sitemap(s) recursively to get all content URLs
            relative_paths = await self._aprocess_sitemap(soup_info, self.base_url)
            if not relative_paths and self.show_progress:
                warnings.warn(
                    f"No content URLs found in sitemap at {self.web_paths[0]}"
                )

            urls = [urljoin(self.base_url, path) for path in relative_paths]

            # Use base class's ascrape_all for efficient parallel fetching
            soup_infos = await self.ascrape_all(urls)

            for soup_info, url in zip(soup_infos, urls):
                maybe_doc = self._get_document(soup_info, url)
                if maybe_doc is not None:
                    yield maybe_doc

    def _get_document(
        self, soup: Any, custom_url: Optional[str] = None
    ) -> Optional[Document]:
        """Fetch content from page and return Document."""
        page_content_raw = soup.find(self.content_selector)
        if not page_content_raw:
            return None
        content = page_content_raw.get_text(separator="\n").strip()
        title_if_exists = page_content_raw.find("h1")
        title = title_if_exists.text if title_if_exists else ""
        metadata = {"source": custom_url or self.web_path, "title": title}
        return Document(page_content=content, metadata=metadata)

    def _get_paths(self, soup: Any) -> List[str]:
        """Fetch all relative paths in the sitemap."""
        return [urlparse(loc.text).path for loc in soup.find_all("loc")]
