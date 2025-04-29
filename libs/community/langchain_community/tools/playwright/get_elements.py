from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Optional, Sequence, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field

from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)

if TYPE_CHECKING:
    from playwright.async_api import Page as AsyncPage
    from playwright.sync_api import Page as SyncPage


class GetElementsToolInput(BaseModel):
    """Input for GetElementsTool."""

    selector: str = Field(
        ...,
        description="CSS selector, such as '*', 'div', 'p', 'a', #id, .classname",
    )
    attributes: List[str] = Field(
        default_factory=lambda: ["innerText"],
        description="Set of attributes to retrieve for each element",
    )


async def _aget_elements(
    page: AsyncPage, selector: str, attributes: Sequence[str]
) -> List[dict]:
    """Get elements matching the given CSS selector."""
    elements = await page.query_selector_all(selector)
    results = []
    for element in elements:
        result = {}
        for attribute in attributes:
            if attribute == "innerText":
                val: Optional[str] = await element.inner_text()
            else:
                val = await element.get_attribute(attribute)
            if val is not None and val.strip() != "":
                result[attribute] = val
        if result:
            results.append(result)
    return results


def _get_elements(
    page: SyncPage, selector: str, attributes: Sequence[str]
) -> List[dict]:
    """Get elements matching the given CSS selector."""
    elements = page.query_selector_all(selector)
    results = []
    for element in elements:
        result = {}
        for attribute in attributes:
            if attribute == "innerText":
                val: Optional[str] = element.inner_text()
            else:
                val = element.get_attribute(attribute)
            if val is not None and val.strip() != "":
                result[attribute] = val
        if result:
            results.append(result)
    return results


class GetElementsTool(BaseBrowserTool):
    """Tool for getting elements in the current web page matching a CSS selector."""

    name: str = "get_elements"
    description: str = (
        "Retrieve elements in the current web page matching the given CSS selector"
    )
    args_schema: Type[BaseModel] = GetElementsToolInput

    def _run(
        self,
        selector: str,
        attributes: Sequence[str] = ["innerText"],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)
        # Navigate to the desired webpage before using this tool
        results = _get_elements(page, selector, attributes)
        return json.dumps(results, ensure_ascii=False)

    async def _arun(
        self,
        selector: str,
        attributes: Sequence[str] = ["innerText"],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)
        # Navigate to the desired webpage before using this tool
        results = await _aget_elements(page, selector, attributes)
        return json.dumps(results, ensure_ascii=False)
