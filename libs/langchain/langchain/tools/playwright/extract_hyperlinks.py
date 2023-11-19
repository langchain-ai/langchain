from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional, Type

from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import aget_current_page, get_current_page

if TYPE_CHECKING:
    pass


class ExtractHyperlinksToolInput(BaseModel):
    """Input for ExtractHyperlinksTool."""

    absolute_urls: bool = Field(
        default=False,
        description="Return absolute URLs instead of relative URLs",
    )


class ExtractHyperlinksTool(BaseBrowserTool):
    """Extract all hyperlinks on the page."""

    name: str = "extract_hyperlinks"
    description: str = "Extract all hyperlinks on the current webpage"
    args_schema: Type[BaseModel] = ExtractHyperlinksToolInput

    @root_validator
    def check_bs_import(cls, values: dict) -> dict:
        """Check that the arguments are valid."""
        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'beautifulsoup4' package is required to use this tool."
                " Please install it with 'pip install beautifulsoup4'."
            )
        return values

    @staticmethod
    def scrape_page(page: Any, html_content: str, absolute_urls: bool) -> str:
        from urllib.parse import urljoin

        from bs4 import BeautifulSoup

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, "lxml")

        # Find all the anchor elements and extract their href attributes
        anchors = soup.find_all("a")
        if absolute_urls:
            base_url = page.url
            links = [urljoin(base_url, anchor.get("href", "")) for anchor in anchors]
        else:
            links = [anchor.get("href", "") for anchor in anchors]
        # Return the list of links as a JSON string
        return json.dumps(links)

    def _run(
        self,
        absolute_urls: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)
        html_content = page.content()
        return self.scrape_page(page, html_content, absolute_urls)

    async def _arun(
        self,
        absolute_urls: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)
        html_content = await page.content()
        return self.scrape_page(page, html_content, absolute_urls)
