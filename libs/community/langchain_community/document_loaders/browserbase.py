from typing import Iterator, List, Optional, Tuple, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class BrowserbaseLoader(BaseLoader):
    """Load pre-rendered web pages using a headless browser hosted on Browserbase.

    Depends on `browserbase` package.
    Get your API key from https://browserbase.com
    """

    def __init__(
        self,
        urls: Union[List[str], Tuple[str, ...]],
        *,
        api_key: Optional[str] = None,
        text_content: bool = False,
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
