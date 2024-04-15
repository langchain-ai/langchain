from typing import Iterator, List

from browserbase import Browserbase
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class BrowserbaseLoader(BaseLoader):
    """Create new Browserbase loader"""

    def __init__(
        self, browserbase: Browserbase, urls: List[str], text_content: str = False
    ):
        self.browserbase = browserbase
        self.urls = urls
        self.text_content = text_content

    def lazy_load(self) -> Iterator[Document]:
        """Load pages from URLs"""
        pages = self.browserbase.load_urls(self.urls, self.text_content)

        for i, page in enumerate(pages):
            yield Document(
                page_content=page,
                metadata={
                    "url": self.urls[i],
                },
            )
