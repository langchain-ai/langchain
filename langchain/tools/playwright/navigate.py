from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field

from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import (
    get_current_page,
)


class NavigateToolInput(BaseModel):
    """Input for NavigateToolInput."""

    url: str = Field(..., description="url to navigate to")


class NavigateTool(BaseBrowserTool):
    name: str = "navigate_browser"
    description: str = "Navigate a browser to the specified URL"
    args_schema: Type[BaseModel] = NavigateToolInput

    async def _arun(self, url: str) -> str:
        """Use the tool."""
        page = await get_current_page(self.browser)
        response = await page.goto(url)
        status = response.status if response else "unknown"
        return f"Navigating to {url} returned status code {status}"
