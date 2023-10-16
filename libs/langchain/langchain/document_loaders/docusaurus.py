"""Load Documents from Docusarus Documentation"""
from typing import Any, List, Optional

from bs4 import BeautifulSoup

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class DocusaurusLoader(BaseLoader):
    """
    Loader that leverages the SitemapLoader to loop through the generated pages of a
    Docusaurus Documentation website and extracts the content by looking for specific
    HTML tags. By default, the parser searches for the main content of the Docusaurus
    page, which is normally the <article>. You also have the option to define your own
    custom HTML tags by providing them as a list, for example: ["div", ".main", "a"].
    """

    def __init__(
        self,
        url: str,
        custom_html_tags: List[str] = ["main article"],
        **kwargs: Optional[Any],
    ):
        """
        Initialize DocusaurusLoader
        Args:
            url: The base URL of the Docusaurus website.
            custom_html_tags: Optional custom html tags to extract content from pages.
            kwargs: Additional args to extend the underlying SitemapLoader, for example:
                filter_urls, blocksize, meta_function, is_local, continue_on_failure
        """
        self.url = url
        self.custom_html_tags = custom_html_tags
        self.kwargs = kwargs

    def load(self) -> List[Document]:
        """Load documents."""
        from langchain.document_loaders.sitemap import SitemapLoader

        if not self.kwargs.get("is_local"):
            self.url = f"{self.url}/sitemap.xml"

        loader = SitemapLoader(
            web_path=self.url,
            parsing_function=self._parsing_function,
            **self.kwargs,
        )
        return loader.load()

    def _parsing_function(self, content: BeautifulSoup) -> str:
        """Parses specific elements from a Docusarus page."""
        relevant_elements = content.select(",".join(self.custom_html_tags))

        for element in relevant_elements:
            if element not in relevant_elements:
                element.decompose()

        return str(content.get_text())
