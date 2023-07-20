from typing import Iterator, List

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class HTML2TextLoader(BaseLoader):
    """Loader for websites using html2text for Markdown output."""

    def __init__(self, urls: List[str]):
        self.web_paths = urls

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load Documents from urls."""

        try:
            import html2text
        except ImportError:
            raise ValueError(
                """html2text package not found, please 
                install it with `pip install html2text`"""
            )

        # Create an html2text.HTML2Text object and override some properties
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        # Process each url
        for path in self.web_paths:
            response = requests.get(path)
            text = h.handle(response.text)
            metadata = {"source": path}
            yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        """Load Documents from table."""
        return list(self.lazy_load())
