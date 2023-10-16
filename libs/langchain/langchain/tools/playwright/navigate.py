from __future__ import annotations

from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)


class NavigateToolInput(BaseModel):
    """Input for NavigateToolInput."""

    url: str = Field(..., description="url to navigate to")


class NavigateTool(BaseBrowserTool):
    """Tool for navigating a browser to a URL."""

    name: str = "navigate_browser"
    description: str = "Navigate a browser to the specified URL"
    args_schema: Type[BaseModel] = NavigateToolInput

    def _run(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)
        response = page.goto(url)
        status = response.status if response else "unknown"
        return f"Navigating to {url} returned status code {status}"

    async def _arun(
        self,
        url: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)
        response = await page.goto(url)
        status = response.status if response else "unknown"
        return f"Navigating to {url} returned status code {status}"
