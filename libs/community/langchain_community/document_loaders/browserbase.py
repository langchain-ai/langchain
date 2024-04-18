from typing import Iterator, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.utils import get_from_env

from langchain_community.document_loaders.base import BaseLoader


class BrowserbaseLoader(BaseLoader):
    """Load pre-rendered web pages using a headless browser hosted on Browserbase.

    Depends on `browserbase` package.
    Get your API key from https://browserbase.com
    """

    def __init__(
        self,
        urls: Sequence[str],
        api_key: Optional[str] = None,
        text_content: str = False,
    ):
        self.urls = urls
        self.text_content = text_content

        try:
            from browserbase import Browserbase
        except ImportError:
            raise ImportError(
                "You must run "
                "`pip install --upgrade "
                "browserbase` "
                "to use the Browserbase loader."
            )

        api_key = api_key or get_from_env("api_key", "BROWSERBASE_API_KEY")
        self.browserbase = Browserbase(api_key=api_key)

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
