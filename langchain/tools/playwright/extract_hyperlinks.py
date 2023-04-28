from __future__ import annotations

import json
from typing import TYPE_CHECKING, Type

from pydantic import BaseModel, Field, root_validator

from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import get_current_page

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
    def check_args(cls, values: dict) -> dict:
        """Check that the arguments are valid."""
        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError:
            raise ValueError(
                "The 'beautifulsoup4' package is required to use this tool."
                " Please install it with 'pip install beautifulsoup4'."
            )
        return values

    async def _arun(self, absolute_urls: bool = False) -> str:
        """Use the tool."""
        from urllib.parse import urljoin

        from bs4 import BeautifulSoup

        page = await get_current_page(self.browser)
        html_content = await page.content()

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
