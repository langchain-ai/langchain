from __future__ import annotations

from typing import Optional, Type

from langchain_core.pydantic_v1 import BaseModel

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import aget_current_page, get_current_page


class CurrentWebPageTool(BaseBrowserTool):
    """Tool for getting the URL of the current webpage."""

    name: str = "current_webpage"
    description: str = "Returns the URL of the current page"
    args_schema: Type[BaseModel] = BaseModel

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)
        return str(page.url)

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)
        return str(page.url)
