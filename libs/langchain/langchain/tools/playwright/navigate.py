from __future__ import annotations

from typing import Optional, Type
from urllib.parse import urlparse

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.tools.playwright.base import BaseBrowserTool
from langchain.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)


class NavigateToolInput(BaseModel):
    """Input for NavigateToolInput."""

    url: str = Field(..., description="url to navigate to")

    @validator("url")
    def validate_url_scheme(cls, url: str) -> str:
        """Check that the URL scheme is valid."""
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError("URL scheme must be 'http' or 'https'")
        return url


class NavigateTool(BaseBrowserTool):
    """Tool for navigating a browser to a URL.

    **Security Note**: This tool provides code to control web-browser navigation.

        This tool can navigate to any URL, including internal network URLs, and
        URLs exposed on the server itself.

        However, if exposing this tool to end-users, consider limiting network
        access to the server that hosts the agent.

        By default, the URL scheme has been limited to 'http' and 'https' to
        prevent navigation to local file system URLs (or other schemes).

        If access to the local file system is required, consider creating a custom
        tool or providing a custom args_schema that allows the desired URL schemes.

        See https://python.langchain.com/docs/security for more information.
    """

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
