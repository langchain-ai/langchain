import asyncio
from typing import Any, Iterator, List, Optional
from urllib.parse import urljoin, urlparse

from langchain_core.documents import Document

from langchain_community.document_loaders.web_base import WebBaseLoader


class GitbookLoader(WebBaseLoader):
    """Load `GitBook` data.

    1. load from either a single page, or
    2. load all (relative) paths in the navbar.
    """

    def __init__(
        self,
        web_page: str,
        load_all_paths: bool = False,
        base_url: Optional[str] = None,
        content_selector: str = "main",
        continue_on_failure: bool = False,
        show_progress: bool = True,
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
        """
        self.base_url = base_url or web_page
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]
        if load_all_paths:
            # set web_path to the sitemap if we want to crawl all paths
            web_page = f"{self.base_url}/sitemap.xml"
        super().__init__(
            web_paths=(web_page,),
            continue_on_failure=continue_on_failure,
            show_progress=show_progress,
        )
        self.load_all_paths = load_all_paths
        self.content_selector = content_selector

    def lazy_load(self) -> Iterator[Document]:
        """Fetch text from one single GitBook page."""
        if self.load_all_paths:
            soup_info = self.scrape()
            relative_paths = self._get_paths(soup_info)
            urls = [urljoin(self.base_url, path) for path in relative_paths]
            soup_infos = asyncio.run(self.scrape_all(urls))
            for soup_info, url in zip(soup_infos, urls):
                doc = self._get_document(soup_info, url)
                if doc:
                    yield doc

        else:
            soup_info = self.scrape()
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
