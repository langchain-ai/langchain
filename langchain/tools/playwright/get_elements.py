from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Optional, Sequence, Type

from pydantic import BaseModel, Field

from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import get_current_page

if TYPE_CHECKING:
    from playwright.async_api import Page as AsyncPage


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


async def _get_elements(
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


class GetElementsTool(BaseBrowserTool):
    name: str = "get_elements"
    description: str = (
        "Retrieve elements in the current web page matching the given CSS selector"
    )
    args_schema: Type[BaseModel] = GetElementsToolInput

    async def _arun(
        self, selector: str, attributes: Sequence[str] = ["innerText"]
    ) -> str:
        """Use the tool."""
        page = await get_current_page(self.browser)
        # Navigate to the desired webpage before using this tool
        results = await _get_elements(page, selector, attributes)
        return json.dumps(results)
