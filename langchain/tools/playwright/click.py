from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field

from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import (
    get_current_page,
)


class ClickToolInput(BaseModel):
    """Input for ClickTool."""

    selector: str = Field(..., description="CSS selector for the element to click")


class ClickTool(BaseBrowserTool):
    name: str = "click_element"
    description: str = "Click on an element with the given CSS selector"
    args_schema: Type[BaseModel] = ClickToolInput

    async def _arun(self, selector: str) -> str:
        """Use the tool."""
        page = await get_current_page(self.browser)
        # Navigate to the desired webpage before using this tool
        await page.click(selector)
        return f"Clicked element '{selector}'"
