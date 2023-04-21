from __future__ import annotations

from typing import Type

from pydantic import BaseModel

from langchain.tools.browser.base import BaseBrowserTool
from langchain.tools.browser.utils import (
    get_current_page,
)


class CurrentPageTool(BaseBrowserTool):
    name: str = "current_page"
    description: str = "Returns the URL of the current page"
    args_schema: Type[BaseModel] = BaseModel

    async def _arun(self) -> str:
        """Use the tool."""
        page = await get_current_page(self.browser)
        return str(page.url)
