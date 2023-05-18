"""Loader that uses bs4 to load HTML files, enriching metadata with page title."""

import logging
from typing import Any, Dict, Iterator, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob

logger = logging.getLogger(__name__)


class BS4HTMLParser(BaseBlobParser):
    """Parser that uses beautiful soup to parse HTML files."""

    def __init__(
        self,
        *,
        features: str = "lxml",
        get_text_separator: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize a bs4 based HTML parser."""
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ValueError(
                "beautifulsoup4 package not found, please install it with "
                "`pip install beautifulsoup4`"
            )

        self.bs_kwargs = {"features": features, **kwargs}
        self.get_text_separator = get_text_separator

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Load HTML document into document objects."""
        from bs4 import BeautifulSoup

        with blob.as_bytes_io() as f:
            soup = BeautifulSoup(f, **self.bs_kwargs)

        text = soup.get_text(self.get_text_separator)

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        metadata: Dict[str, Union[str, None]] = {
            "source": blob.source,
            "title": title,
        }
        yield Document(page_content=text, metadata=metadata)
