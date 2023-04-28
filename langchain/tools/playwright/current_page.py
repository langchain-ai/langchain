from __future__ import annotations

from typing import Optional, Type

from pydantic import BaseModel

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import (
    get_current_page,
)


class CurrentWebPageTool(BaseBrowserTool):
    name: str = "current_webpage"
    description: str = "Returns the URL of the current page"
    args_schema: Type[BaseModel] = BaseModel

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        page = await get_current_page(self.browser)
        return str(page.url)
