import os
from typing import Iterator, List
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class BrowserbaseLoader(BaseLoader):
    """Create new Browserbase loader"""

    def __init__(
        self, urls: List[str], api_key: str = os.environ["BROWSERBASE_KEY"], text_content: str = False
    ):
        self.urls = urls
        self.api_key = api_key
        self.text_content = text_content

    def lazy_load(self) -> Iterator[Document]:
        """Load pages from URLs"""
        try:
            from browserbase import Browserbase
        except ImportError:
            raise ImportError(
                "You must run "
                "`pip install --upgrade "
                "browserbase` "
                "to use the Browserbase loader."
            )

        browser = Browserbase(api_key=self.api_key)
        pages = browser.load_urls(self.urls, self.text_content)

        for i, page in enumerate(pages):
            yield Document(
                page_content=page,
                metadata={
                    "url": self.urls[i],
                },
            )
