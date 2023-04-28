from __future__ import annotations

from typing import Type

from pydantic import BaseModel

from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import (
    get_current_page,
)


class NavigateBackTool(BaseBrowserTool):
    """Navigate back to the previous page in the browser history."""

    name: str = "previous_webpage"
    description: str = "Navigate back to the previous page in the browser history"
    args_schema: Type[BaseModel] = BaseModel

    async def _arun(self) -> str:
        """Use the tool."""
        page = await get_current_page(self.browser)
        response = await page.go_back()

        if response:
            return (
                f"Navigated back to the previous page with URL '{response.url}'."
                " Status code {response.status}"
            )
        else:
            return "Unable to navigate back; no previous page in the history"
