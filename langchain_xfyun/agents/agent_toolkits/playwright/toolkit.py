"""Playwright web browser toolkit."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Type, cast

from langchain_xfyun.agents.agent_toolkits.base import BaseToolkit
from langchain_xfyun.pydantic_v1 import Extra, root_validator
from langchain_xfyun.tools.base import BaseTool
from langchain_xfyun.tools.playwright.base import (
    BaseBrowserTool,
    lazy_import_playwright_browsers,
)
from langchain_xfyun.tools.playwright.click import ClickTool
from langchain_xfyun.tools.playwright.current_page import CurrentWebPageTool
from langchain_xfyun.tools.playwright.extract_hyperlinks import ExtractHyperlinksTool
from langchain_xfyun.tools.playwright.extract_text import ExtractTextTool
from langchain_xfyun.tools.playwright.get_elements import GetElementsTool
from langchain_xfyun.tools.playwright.navigate import NavigateTool
from langchain_xfyun.tools.playwright.navigate_back import NavigateBackTool

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.sync_api import Browser as SyncBrowser
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from playwright.async_api import Browser as AsyncBrowser
        from playwright.sync_api import Browser as SyncBrowser
    except ImportError:
        pass


class PlayWrightBrowserToolkit(BaseToolkit):
    """Toolkit for PlayWright browser tools."""

    sync_browser: Optional["SyncBrowser"] = None
    async_browser: Optional["AsyncBrowser"] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator
    def validate_imports_and_browser_provided(cls, values: dict) -> dict:
        """Check that the arguments are valid."""
        lazy_import_playwright_browsers()
        if values.get("async_browser") is None and values.get("sync_browser") is None:
            raise ValueError("Either async_browser or sync_browser must be specified.")
        return values

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tool_classes: List[Type[BaseBrowserTool]] = [
            ClickTool,
            NavigateTool,
            NavigateBackTool,
            ExtractTextTool,
            ExtractHyperlinksTool,
            GetElementsTool,
            CurrentWebPageTool,
        ]

        tools = [
            tool_cls.from_browser(
                sync_browser=self.sync_browser, async_browser=self.async_browser
            )
            for tool_cls in tool_classes
        ]
        return cast(List[BaseTool], tools)

    @classmethod
    def from_browser(
        cls,
        sync_browser: Optional[SyncBrowser] = None,
        async_browser: Optional[AsyncBrowser] = None,
    ) -> PlayWrightBrowserToolkit:
        """Instantiate the toolkit."""
        # This is to raise a better error than the forward ref ones Pydantic would have
        lazy_import_playwright_browsers()
        return cls(sync_browser=sync_browser, async_browser=async_browser)
