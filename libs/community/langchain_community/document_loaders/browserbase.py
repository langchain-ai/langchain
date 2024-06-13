from typing import Iterator, Optional, Sequence

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class BrowserbaseLoader(BaseLoader):
    """Load pre-rendered web pages using a headless browser hosted on Browserbase.

    Depends on `browserbase` package.
    Get your API key from https://browserbase.com
    """

    def __init__(
        self,
        urls: Sequence[str],
        text_content: bool = False,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        proxy: Optional[bool] = None,
    ):
        self.urls = urls
        self.text_content = text_content
        self.session_id = session_id
        self.proxy = proxy

        try:
            from browserbase import Browserbase
        except ImportError:
            raise ImportError(
                "You must run "
                "`pip install --upgrade "
                "browserbase` "
                "to use the Browserbase loader."
            )

        self.browserbase = Browserbase(api_key, project_id)

    def lazy_load(self) -> Iterator[Document]:
        """Load pages from URLs"""
        pages = self.browserbase.load_urls(
            self.urls, self.text_content, self.session_id, self.proxy
        )

        for i, page in enumerate(pages):
            yield Document(
                page_content=page,
                metadata={
                    "url": self.urls[i],
                },
            )
