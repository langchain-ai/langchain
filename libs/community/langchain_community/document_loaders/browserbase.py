from typing import Any, Dict, Iterator, Optional, Sequence

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class BrowserbaseLoader(BaseLoader):
    """Load pre-rendered web pages using a headless browser hosted on Browserbase.

    Depends on `browserbase` and `playwright` packages.
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
        self.project_id = project_id
        self.proxy = proxy

        try:
            from browserbase import Browserbase
        except ImportError:
            raise ImportError(
                "You must run "
                "`pip install --upgrade "
                "browserbase playwright` "
                "to use the Browserbase loader."
            )

        self.browserbase = Browserbase(api_key=api_key)

    def lazy_load(self) -> Iterator[Document]:
        """Load pages from URLs"""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "playwright is required for BrowserbaseLoader. "
                "Please run `pip install --upgrade playwright`."
            )

        for url in self.urls:
            with sync_playwright() as playwright:
                # Create or use existing session
                if self.session_id:
                    session = self.browserbase.sessions.retrieve(id=self.session_id)
                else:
                    if not self.project_id:
                        raise ValueError("project_id is required to create a session")
                    session_params: Dict[str, Any] = {"project_id": self.project_id}
                    if self.proxy is not None:
                        session_params["proxy"] = bool(self.proxy)
                    session = self.browserbase.sessions.create(**session_params)

                # Connect to the remote session
                browser = playwright.chromium.connect_over_cdp(session.connect_url)
                context = browser.contexts[0]
                page = context.pages[0]

                # Navigate to URL and get content
                page.goto(url)
                # Get content based on the text_content flag
                if self.text_content:
                    page_text = page.inner_text("body")
                    content = str(page_text)
                else:
                    page_html = page.content()
                    content = str(page_html)

                # Close browser
                page.close()
                browser.close()

                yield Document(
                    page_content=content,
                    metadata={
                        "url": url,
                    },
                )
