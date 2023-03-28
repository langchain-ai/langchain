"""Loader that loads GitBook."""
from typing import Any, List, Optional
from urllib.parse import urlparse

from langchain.docstore.document import Document
from langchain.document_loaders.web_base import WebBaseLoader


class GitbookLoader(WebBaseLoader):
    """Load GitBook data.

    1. load from either a single page, or
    2. load all (relative) paths in the navbar.
    """

    def __init__(
        self,
        web_page: str,
        load_all_paths: bool = False,
        base_url: Optional[str] = None,
    ):
        """Initialize with web page and whether to load all paths.

        Args:
            web_page: The web page to load or the starting point from where
                relative paths are discovered.
            load_all_paths: If set to True, all relative paths in the navbar
                are loaded instead of only `web_page`.
            base_url: If `load_all_paths` is True, the relative paths are
                appended to this base url. Defaults to `web_page` if not set.
        """
        self.base_url = base_url or web_page
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]
        if load_all_paths:
            # set web_path to the sitemap if we want to crawl all paths
            web_paths = f"{self.base_url}/sitemap.xml"
        else:
            web_paths = web_page
        super().__init__(web_paths)
        self.load_all_paths = load_all_paths

    def load(self) -> List[Document]:
        """Fetch text from one single GitBook page."""
        if self.load_all_paths:
            soup_info = self.scrape()
            relative_paths = self._get_paths(soup_info)
            documents = []
            for path in relative_paths:
                url = self.base_url + path
                print(f"Fetching text from {url}")
                soup_info = self._scrape(url)
                documents.append(self._get_document(soup_info, url))
            return documents
        else:
            soup_info = self.scrape()
            return [self._get_document(soup_info, self.web_path)]

    def _get_document(self, soup: Any, custom_url: Optional[str] = None) -> Document:
        """Fetch content from page and return Document."""
        page_content_raw = soup.find("main")
        content = page_content_raw.get_text(separator="\n").strip()
        title_if_exists = page_content_raw.find("h1")
        title = title_if_exists.text if title_if_exists else ""
        metadata = {"source": custom_url or self.web_path, "title": title}
        return Document(page_content=content, metadata=metadata)

    def _get_paths(self, soup: Any) -> List[str]:
        """Fetch all relative paths in the navbar."""
        return [urlparse(loc.text).path for loc in soup.find_all("loc")]
