"""Loader that loads GitBook."""
from typing import Any, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.web_base import WebBaseLoader

class GitbookLoader(WebBaseLoader):
    """Load GitBook data from either a single page or all (relative) paths in the navbar."""

    def load(self, custom_web_path: Optional[str] = None) -> List[Document]:
        """Fetch text from one single GitBook page"""
        soup_info = self.scrape(custom_web_path)
        url = custom_web_path if custom_web_path else self.web_path
        return [self._get_document(soup_info, url)]

    def load_from_all_paths(self) -> List[Document]:
        """ Fetch text from all pages in the navbar
            Make sure the initialized web_path is the root of the GitBook
        """
        soup_info = self.scrape()
        relative_paths = self._get_paths(soup_info)
        documents = []
        for path in relative_paths:
            url = self.web_path + path
            print(f"Fetching text from {url}")
            documents += self.load(url)
        return documents

    def _get_document(self, soup: Any, custom_url: Optional[str] = None) -> Document:
        """Fetch content from page and return Document"""
        page_content_raw = soup.find("main")
        content = page_content_raw.get_text(separator='\n').strip()
        title_if_exists = page_content_raw.find("h1")
        title = title_if_exists.text if title_if_exists else ""
        metadata = {
            "source": custom_url if custom_url else  self.web_path, 
            "title": title
        }
        return Document(page_content=content, metadata=metadata)

    def _get_paths(self, soup: Any) -> List[str]:
        """Fetch all relative paths in the navbar"""
        nav = soup.find("nav")
        links = nav.findAll("a")
        # only return relative links
        return [link.get("href") for link in links
                if link.get("href")[0] == "/"]
