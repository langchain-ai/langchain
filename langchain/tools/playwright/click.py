from __future__ import annotations

from typing import Optional, Type

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)


class ClickToolInput(BaseModel):
    """Input for ClickTool."""

    selector: str = Field(..., description="CSS selector for the element to click")


class ClickTool(BaseBrowserTool):
    name: str = "click_element"
    description: str = "Click on an element with the given CSS selector"
    args_schema: Type[BaseModel] = ClickToolInput

    def _run(
        self,
        selector: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)
        # Navigate to the desired webpage before using this tool
        try:
            page.click(selector)
            return f"Clicked element '{selector}'"
        except Exception as e:
            return f"Error '{e}'"

    async def _arun(
        self,
        selector: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)
        # Navigate to the desired webpage before using this tool
        try:
            await page.click(selector)
            return f"Clicked element '{selector}'"
        except Exception as e:
            return f"Error '{e}'"
