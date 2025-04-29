import warnings
from typing import Any, AsyncIterator, Iterator, List, Optional, Set, Union
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.web_base import WebBaseLoader


class GitbookLoader(BaseLoader):
    """Load `GitBook` data.

    1. load from either a single page, or
    2. load all (relative) paths in the sitemap, handling nested sitemap indexes.

    When `load_all_paths=True`, the loader parses XML sitemaps and requires the
    `lxml` package to be installed (`pip install lxml`).
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
        allowed_domains: Optional[Set[str]] = None,
    ):
        """Initialize with web page and whether to load all paths.

        Args:
            web_page: The web page to load or the starting point from where
                relative paths are discovered.
            load_all_paths: If set to True, all relative paths in the navbar
                are loaded instead of only `web_page`. Requires `lxml` package.
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
            allowed_domains: Optional set of allowed domains to fetch from.
                If None (default), the loader will restrict crawling to the domain
                of the `web_page` URL to prevent potential SSRF vulnerabilities.
                Provide an explicit set (e.g., {"example.com", "docs.example.com"})
                to allow crawling across multiple domains. Use with caution in
                server environments where users might control the input URLs.
        """
        self.base_url = base_url or web_page
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

        self.web_page = web_page
        self.load_all_paths = load_all_paths
        self.content_selector = content_selector
        self.continue_on_failure = continue_on_failure
        self.show_progress = show_progress
        self.allowed_domains = allowed_domains

        # If allowed_domains is not specified, extract domain from web_page as default
        if self.allowed_domains is None:
            initial_domain = urlparse(web_page).netloc
            if initial_domain:
                self.allowed_domains = {initial_domain}

        # Determine the starting URL (either a sitemap or a direct page)
        if load_all_paths:
            self.start_url = sitemap_url or f"{self.base_url}/sitemap.xml"
        else:
            self.start_url = web_page

        # Validate the start_url is allowed
        if not self._is_url_allowed(self.start_url):
            raise ValueError(
                f"Domain in {self.start_url} is not in the allowed domains list: "
                f"{self.allowed_domains}"
            )

    def _is_url_allowed(self, url: str) -> bool:
        """Check if a URL has an allowed scheme and domain."""
        # It's assumed self.allowed_domains is always set by __init__
        # either explicitly or derived from web_page. If it's somehow still
        # None here, it indicates an initialization issue, so denying is safer.
        if self.allowed_domains is None:
            return False  # Should not happen if init worked

        try:
            parsed = urlparse(url)

            # 1. Validate scheme (Minimal Enhancement)
            if parsed.scheme not in ("http", "https"):
                return False

            # 2. Validate domain (Existing logic - handles suffix correctly)
            # Ensure netloc is not empty before checking membership
            if not parsed.netloc:
                return False
            return parsed.netloc in self.allowed_domains
        except Exception:  # Catch potential urlparse errors
            return False

    def _safe_add_url(
        self, url_list: List[str], url: str, url_type: str = "URL"
    ) -> bool:
        """Safely add a URL to a list if it's from an allowed domain.

        Args:
            url_list: The list to add the URL to
            url: The URL to add
            url_type: Type of URL for warning message (e.g., "sitemap", "content")

        Returns:
            bool: True if URL was added, False if skipped
        """
        if self._is_url_allowed(url):
            url_list.append(url)
            return True
        else:
            warnings.warn(f"Skipping disallowed {url_type} URL: {url}")
            return False

    def _create_web_loader(self, url_or_urls: Union[str, List[str]]) -> WebBaseLoader:
        """Create a new WebBaseLoader instance for the given URL(s).

        This ensures each operation gets its own isolated WebBaseLoader.
        """
        return WebBaseLoader(
            web_path=url_or_urls,
            continue_on_failure=self.continue_on_failure,
            show_progress=self.show_progress,
        )

    def _is_sitemap_index(self, soup: BeautifulSoup) -> bool:
        """Check if the soup contains a sitemap index."""
        return soup.find("sitemapindex") is not None

    def _extract_sitemap_urls(self, soup: BeautifulSoup) -> List[str]:
        """Extract sitemap URLs from a sitemap index."""
        sitemap_tags = soup.find_all("sitemap")
        urls: List[str] = []
        for sitemap in sitemap_tags:
            loc = sitemap.find("loc")
            if loc and loc.text:
                self._safe_add_url(urls, loc.text, "sitemap")
        return urls

    def _process_sitemap(
        self,
        soup: BeautifulSoup,
        processed_urls: Set[str],
        web_loader: Optional[WebBaseLoader] = None,
    ) -> List[str]:
        """Process a sitemap, handling both direct content URLs and sitemap indexes.

        Args:
            soup: The BeautifulSoup object of the sitemap
            processed_urls: Set of already processed URLs to avoid cycles
            web_loader: WebBaseLoader instance to reuse for all requests,
                created if None
        """
        # Create a loader if not provided
        if web_loader is None:
            web_loader = self._create_web_loader(self.start_url)

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
                    # Temporarily override the web_path of the loader
                    original_web_paths = web_loader.web_paths
                    web_loader.web_paths = [sitemap_url]

                    # Reuse the same loader for the next sitemap,
                    # explicitly use lxml-xml
                    sitemap_soup = web_loader.scrape(parser="lxml-xml")

                    # Restore original web_paths
                    web_loader.web_paths = original_web_paths

                    # Recursive call with the same loader
                    content_urls = self._process_sitemap(
                        sitemap_soup, processed_urls, web_loader
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

    async def _aprocess_sitemap(
        self,
        soup: BeautifulSoup,
        base_url: str,
        processed_urls: Set[str],
        web_loader: Optional[WebBaseLoader] = None,
    ) -> List[str]:
        """Async version of _process_sitemap.

        Args:
            soup: The BeautifulSoup object of the sitemap
            base_url: The base URL for relative paths
            processed_urls: Set of already processed URLs to avoid cycles
            web_loader: WebBaseLoader instance to reuse for all requests,
                created if None
        """
        # Create a loader if not provided
        if web_loader is None:
            web_loader = self._create_web_loader(self.start_url)

        # If it's a sitemap index, recursively process each sitemap URL
        if self._is_sitemap_index(soup):
            sitemap_urls = self._extract_sitemap_urls(soup)
            all_content_urls = []

            # Filter out already processed URLs
            new_urls = [url for url in sitemap_urls if url not in processed_urls]

            if not new_urls:
                return []

            # Update the web_paths of the loader to fetch all sitemaps at once
            original_web_paths = web_loader.web_paths
            web_loader.web_paths = new_urls

            # Use the same WebBaseLoader's ascrape_all for efficient parallel
            # fetching, explicitly use lxml-xml
            soups = await web_loader.ascrape_all(new_urls, parser="lxml-xml")

            # Restore original web_paths
            web_loader.web_paths = original_web_paths

            for sitemap_url, sitemap_soup in zip(new_urls, soups):
                processed_urls.add(sitemap_url)
                try:
                    # Recursive call with the same loader
                    content_urls = await self._aprocess_sitemap(
                        sitemap_soup, base_url, processed_urls, web_loader
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
        if not self.load_all_paths:
            # Simple case: load a single page
            temp_loader = self._create_web_loader(self.web_page)
            soup = temp_loader.scrape()
            doc = self._get_document(soup, self.web_page)
            if doc:
                yield doc
        else:
            # Get initial sitemap using the recursive method
            temp_loader = self._create_web_loader(self.start_url)
            # Explicitly use lxml-xml for parsing the initial sitemap
            soup_info = temp_loader.scrape(parser="lxml-xml")

            # Process sitemap(s) recursively to get all content URLs
            processed_urls: Set[str] = set()
            relative_paths = self._process_sitemap(soup_info, processed_urls)

            if not relative_paths and self.show_progress:
                warnings.warn(f"No content URLs found in sitemap at {self.start_url}")

            # Build full URLs from relative paths
            urls: List[str] = []
            for url in relative_paths:
                # URLs are now already absolute from _get_paths
                self._safe_add_url(urls, url, "content")

            if not urls:
                return

            # Create a loader for content pages
            content_loader = self._create_web_loader(urls)

            # Use WebBaseLoader to fetch all pages
            soup_infos = content_loader.scrape_all(urls)

            for soup_info, url in zip(soup_infos, urls):
                doc = self._get_document(soup_info, url)
                if doc:
                    yield doc

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Asynchronously fetch text from GitBook page(s)."""
        if not self.load_all_paths:
            # Simple case: load a single page asynchronously
            temp_loader = self._create_web_loader(self.web_page)
            soups = await temp_loader.ascrape_all([self.web_page])
            soup_info = soups[0]
            doc = self._get_document(soup_info, self.web_page)
            if doc:
                yield doc
        else:
            # Get initial sitemap - web_loader will be created in _aprocess_sitemap
            temp_loader = self._create_web_loader(self.start_url)
            # Explicitly use lxml-xml for parsing the initial sitemap
            soups = await temp_loader.ascrape_all([self.start_url], parser="lxml-xml")
            soup_info = soups[0]

            # Process sitemap(s) recursively to get all content URLs
            processed_urls: Set[str] = set()
            relative_paths = await self._aprocess_sitemap(
                soup_info, self.base_url, processed_urls
            )

            if not relative_paths and self.show_progress:
                warnings.warn(f"No content URLs found in sitemap at {self.start_url}")

            # Build full URLs from relative paths
            urls: List[str] = []
            for url in relative_paths:
                # URLs are now already absolute from _get_paths
                self._safe_add_url(urls, url, "content")

            if not urls:
                return

            # Create a loader for content pages
            content_loader = self._create_web_loader(urls)

            # Use WebBaseLoader's ascrape_all for efficient parallel fetching
            soup_infos = await content_loader.ascrape_all(urls)

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
        metadata = {"source": custom_url or self.web_page, "title": title}
        return Document(page_content=content, metadata=metadata)

    def _get_paths(self, soup: Any) -> List[str]:
        """Fetch all URLs in the sitemap."""
        urls = []
        for loc in soup.find_all("loc"):
            if loc.text:
                # Instead of extracting just the path, keep the full URL
                # to preserve domain information
                urls.append(loc.text)
        return urls
