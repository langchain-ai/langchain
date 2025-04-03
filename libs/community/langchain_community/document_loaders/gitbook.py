import asyncio
from typing import Any, AsyncIterator, Iterator, List, Optional, Set
from urllib.parse import urljoin, urlparse

from langchain_core.documents import Document

from langchain_community.document_loaders.web_base import WebBaseLoader


class GitbookLoader(WebBaseLoader):
    """Load `GitBook` data.

    1. load from either a single page, or
    2. load all (relative) paths in the navbar.

    Supports both synchronous and asynchronous loading.
    """

    def __init__(
        self,
        web_page: str,
        load_all_paths: bool = False,
        base_url: Optional[str] = None,
        content_selector: str = "main",
        continue_on_failure: bool = False,
        show_progress: bool = True,
        max_concurrent_requests: int = 10,
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
            max_concurrent_requests: Maximum number of concurrent requests when
                using async methods. Default: 10
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
        self._processed_urls: Set[str] = set()
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_requests))

    def _is_sitemap(self, soup: Any) -> bool:
        """Check if the soup contains a sitemap structure."""
        # Check for sitemap namespace or sitemap structure
        return bool(soup.find("urlset")) or bool(soup.find("sitemapindex"))

    def _process_sitemap(self, soup: Any, base_url: str) -> List[str]:
        """Process a sitemap and extract all URLs, including from nested sitemaps."""
        urls = []

        # Check if this is a sitemap index (containing other sitemaps)
        sitemap_index = soup.find("sitemapindex")
        if sitemap_index:
            # Process nested sitemaps in the sitemap index
            for sitemap in soup.find_all("sitemap"):
                loc = sitemap.find("loc")
                if loc and loc.text:
                    relative_url = loc.text
                    # Construct the absolute URL for the nested sitemap
                    nested_url = urljoin(base_url, relative_url)
                    if nested_url not in self._processed_urls:
                        self._processed_urls.add(nested_url)
                        try:
                            # Use XML parser for sitemaps
                            nested_soup = self._scrape(nested_url, parser="xml")
                            # Pass base_url, not the potentially relative nested_url
                            nested_urls = self._process_sitemap(nested_soup, base_url)
                            urls.extend(nested_urls)
                        except Exception as e:
                            if self.continue_on_failure:
                                import warnings

                                warnings.warn(
                                    f"Error processing nested sitemap {nested_url}: {e}"
                                )
                            else:
                                raise
        else:
            # This is a regular sitemap with URLs
            for url_tag in soup.find_all("url"):
                loc = url_tag.find("loc")
                if loc and loc.text:
                    page_url = loc.text
                    if page_url not in self._processed_urls:
                        self._processed_urls.add(page_url)
                        urls.append(page_url)

        return urls

    def lazy_load(self) -> Iterator[Document]:
        """Fetch text from GitBook pages, handling nested sitemaps."""
        if self.load_all_paths:
            self._processed_urls = set()

            # Scrape the initial URL which could be a sitemap
            try:
                # Try with XML parser first for sitemaps
                initial_soup = self._scrape(self.web_path, parser="xml")
            except Exception:
                # Fallback to default parser
                initial_soup = self.scrape()

            if self._is_sitemap(initial_soup):
                # Process sitemap and any nested sitemaps
                urls = self._process_sitemap(initial_soup, self.base_url)
            else:
                # Not a sitemap, treat as regular page with links
                relative_paths = self._get_paths(initial_soup)
                urls = [urljoin(self.base_url, path) for path in relative_paths]

            if urls:
                soup_infos = self.scrape_all(urls)
                for soup_info, url in zip(soup_infos, urls):
                    # Skip if this is another sitemap
                    if self._is_sitemap(soup_info):
                        continue

                    doc = self._get_document(soup_info, url)
                    if doc:
                        yield doc
        else:
            soup_info = self.scrape()
            # Skip if this is a sitemap
            if not self._is_sitemap(soup_info):
                doc = self._get_document(soup_info, self.web_path)
                if doc:
                    yield doc

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
        """Fetch all relative paths in the navbar."""
        return [urlparse(loc.text).path for loc in soup.find_all("loc")]

    async def _ascrape(self, url: str, parser: Optional[str] = None) -> Any:
        """Asynchronously scrape a URL with concurrency control."""
        parser = parser or "html.parser"
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(
                    None, lambda: self._scrape(url, parser=parser)
                )
            except Exception as e:
                if self.continue_on_failure:
                    import warnings

                    warnings.warn(f"Error fetching {url}: {e}")
                    return None
                else:
                    raise

    async def _aprocess_sitemap(self, soup: Any, base_url: str) -> List[str]:
        """Process a sitemap asynchronously and extract all URLs, including from nested
        sitemaps."""
        urls: List[str] = []
        if not soup:
            return urls

        # Check if this is a sitemap index
        sitemap_index = soup.find("sitemapindex")
        if sitemap_index:
            # Process nested sitemaps concurrently
            tasks = []
            for sitemap in soup.find_all("sitemap"):
                loc = sitemap.find("loc")
                if loc and loc.text:
                    relative_url = loc.text
                    nested_url = urljoin(base_url, relative_url)
                    if nested_url not in self._processed_urls:
                        self._processed_urls.add(nested_url)
                        tasks.append(
                            self._aprocess_nested_sitemap(nested_url, base_url)
                        )

            # Wait for all nested sitemap tasks to complete
            nested_results = await asyncio.gather(
                *tasks, return_exceptions=self.continue_on_failure
            )
            for result in nested_results:
                if isinstance(result, Exception):
                    if self.continue_on_failure:
                        import warnings

                        warnings.warn(f"Error processing nested sitemap: {result}")
                elif isinstance(result, list):
                    urls.extend(result)
        else:
            # Regular sitemap with URLs
            for url_tag in soup.find_all("url"):
                loc = url_tag.find("loc")
                if loc and loc.text:
                    page_url = urljoin(base_url, loc.text)
                    if page_url not in self._processed_urls:
                        self._processed_urls.add(page_url)
                        urls.append(page_url)

        return urls

    async def _aprocess_nested_sitemap(
        self, nested_url: str, base_url: str
    ) -> List[str]:
        """Helper to fetch and process a single nested sitemap."""
        try:
            nested_soup = await self._ascrape(nested_url, parser="xml")
            if nested_soup:
                return await self._aprocess_sitemap(nested_soup, base_url)
            return []
        except Exception as e:
            if self.continue_on_failure:
                import warnings

                warnings.warn(f"Error processing nested sitemap {nested_url}: {e}")
                return []
            raise

    async def lazy_aload(self) -> AsyncIterator[Document]:
        """Asynchronously fetch content from GitBook pages."""
        if self.load_all_paths:
            self._processed_urls = set()

            # Fetch and parse the initial sitemap
            try:
                initial_soup = await self._ascrape(self.web_path, parser="xml")
            except Exception:
                if self.continue_on_failure:
                    initial_soup = None
                else:
                    raise

            urls = []
            if initial_soup and self._is_sitemap(initial_soup):
                # Process sitemap(s) to get all content URLs
                urls = await self._aprocess_sitemap(initial_soup, self.base_url)
            elif initial_soup:
                # Not a sitemap, treat as regular page with links
                relative_paths = self._get_paths(initial_soup)
                urls = [urljoin(self.base_url, path) for path in relative_paths]

            if urls:
                # Create tasks for fetching all URLs concurrently
                tasks = [self._afetch_and_process_url(url) for url in urls]

                # Use as_completed to yield results as they arrive, instead of waiting
                # for all
                for future in asyncio.as_completed(tasks):
                    try:
                        doc = await future
                        if doc:
                            yield doc
                    except Exception as e:
                        if self.continue_on_failure:
                            import warnings

                            warnings.warn(f"Error fetching document: {e}")
                        else:
                            raise
        else:
            # Just fetch the single specified page
            try:
                soup_info = await self._ascrape(self.web_path)
                if soup_info and not self._is_sitemap(soup_info):
                    doc = self._get_document(soup_info, self.web_path)
                    if doc:
                        yield doc
            except Exception as e:
                if self.continue_on_failure:
                    import warnings

                    warnings.warn(f"Error fetching single page {self.web_path}: {e}")
                else:
                    raise

    async def _afetch_and_process_url(self, url: str) -> Optional[Document]:
        """Fetch and process a single URL asynchronously."""
        try:
            soup_info = await self._ascrape(url)
            if soup_info and not self._is_sitemap(soup_info):
                return self._get_document(soup_info, url)
        except Exception as e:
            if self.continue_on_failure:
                import warnings

                warnings.warn(f"Error processing URL {url}: {e}")
            else:
                raise
        return None

    async def aload(self) -> List[Document]:  # type: ignore
        """Load all documents asynchronously.

        This method overrides the WebBaseLoader's aload method.
        """
        docs: List[Document] = []
        async for doc in self.alazy_load():
            docs.append(doc)
        return docs

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Asynchronously fetch content from GitBook pages."""
        if self.load_all_paths:
            self._processed_urls = set()

            # Fetch and parse the initial sitemap
            try:
                initial_soup = await self._ascrape(self.web_path, parser="xml")
            except Exception as e:
                if self.continue_on_failure:
                    import warnings

                    warnings.warn(f"Error fetching initial sitemap: {e}")
                    initial_soup = None
                else:
                    raise

            urls = []
            if initial_soup and self._is_sitemap(initial_soup):
                # Process sitemap(s) to get all content URLs
                urls = await self._aprocess_sitemap(initial_soup, self.base_url)
            elif initial_soup:
                # Not a sitemap, treat as regular page with links
                relative_paths = self._get_paths(initial_soup)
                urls = [urljoin(self.base_url, path) for path in relative_paths]

            if urls:
                # Create tasks for fetching all URLs concurrently
                tasks = [self._afetch_and_process_url(url) for url in urls]

                # Use as_completed to yield results as they arrive, instead of waiting
                # for all
                for future in asyncio.as_completed(tasks):
                    try:
                        doc = await future
                        if doc:
                            yield doc
                    except Exception as e:
                        if self.continue_on_failure:
                            import warnings

                            warnings.warn(f"Error fetching document: {e}")
                        else:
                            raise
        else:
            # Simply scrape the single URL
            try:
                soup = await self._ascrape(self.web_path)
                if not self._is_sitemap(soup):
                    doc = self._get_document(soup, self.web_path)
                    if doc:
                        yield doc
            except Exception as e:
                if self.continue_on_failure:
                    import warnings

                    warnings.warn(f"Error fetching single page {self.web_path}: {e}")
                else:
                    raise
