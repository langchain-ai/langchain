"""Load Documents from Docusarus Documentation"""

from typing import Any, List, Optional

from langchain_community.document_loaders.sitemap import SitemapLoader


class DocusaurusLoader(SitemapLoader):
    """Load from Docusaurus Documentation.

    It leverages the SitemapLoader to loop through the generated pages of a
    Docusaurus Documentation website and extracts the content by looking for specific
    HTML tags. By default, the parser searches for the main content of the Docusaurus
    page, which is normally the <article>. You can also define your own
    custom HTML tags by providing them as a list, for example: ["div", ".main", "a"].
    """

    def __init__(
        self,
        url: str,
        custom_html_tags: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize DocusaurusLoader

        Args:
            url: The base URL of the Docusaurus website.
            custom_html_tags: Optional custom html tags to extract content from pages.
            kwargs: Additional args to extend the underlying SitemapLoader, for example:
                filter_urls, blocksize, meta_function, is_local, continue_on_failure
        """
        if not kwargs.get("is_local"):
            url = f"{url}/sitemap.xml"

        self.custom_html_tags = custom_html_tags or ["main article"]

        super().__init__(
            url,
            parsing_function=kwargs.get("parsing_function") or self._parsing_function,
            **kwargs,
        )

    def _parsing_function(self, content: Any) -> str:
        """Parses specific elements from a Docusaurus page."""
        relevant_elements = content.select(",".join(self.custom_html_tags))

        for element in relevant_elements:
            if element not in relevant_elements:
                element.decompose()

        return str(content.get_text())
