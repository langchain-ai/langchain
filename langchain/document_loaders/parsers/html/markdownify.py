"""Load and chunk HTMLs with potential pre-processing to clean the html."""

import re
from typing import Iterator, Tuple

from bs4 import BeautifulSoup

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document

# Regular expression pattern to detect multiple new lines in a row with optional
# whitespace in between
CONSECUTIVE_NEW_LINES = re.compile(r"\n(\s*\n)+", flags=re.UNICODE)


def _get_mini_html(html: str, *, tags_to_remove: Tuple[str, ...] = tuple()) -> str:
    """Clean up HTML tags."""
    # Parse the HTML document using BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Remove all CSS stylesheets
    for stylesheet in soup.find_all("link", rel="stylesheet"):
        stylesheet.extract()

    for tag_to_remove in tags_to_remove:
        # Remove all matching tags
        for tag in soup.find_all(tag_to_remove):
            tag.extract()

    new_html = repr(soup)
    return new_html


def _clean_html(html: str, *, tags_to_remove: Tuple[str, ...] = tuple()) -> str:
    """Clean up HTML and convert to markdown using markdownify."""
    try:
        import markdownify
    except ImportError:
        raise ImportError(
            "The markdownify package is required to parse HTML files. "
            "Please install it with `pip install markdownify`."
        )
    html = _get_mini_html(html, tags_to_remove=tags_to_remove)
    md = markdownify.markdownify(html)
    return CONSECUTIVE_NEW_LINES.sub("\n\n", md).strip()


## PUBLIC API


class MarkdownifyHTMLParser(BaseBlobParser):
    """A blob parser to parse HTML content.."""

    def __init__(
        self,
        tags_to_remove: Tuple[str, ...] = ("svg", "img", "script", "style"),
    ) -> None:
        """Initialize the preprocessor.

        Args:
            tags_to_remove: A tuple of tags to remove from the HTML
        """

        self.tags_to_remove = tags_to_remove

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        yield Document(
            page_content=_clean_html(
                blob.as_string(), tags_to_remove=self.tags_to_remove
            ),
            metadata={"source": blob.source},
        )
