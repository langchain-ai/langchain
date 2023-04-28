"""Playwright web browser toolkit."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Type, cast

from pydantic import Extra, Field, root_validator

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools.base import BaseTool
from langchain.tools.playwright.base import BaseBrowserToolMixin
from langchain.tools.playwright.click import ClickTool
from langchain.tools.playwright.current_page import CurrentWebPageTool
from langchain.tools.playwright.extract_hyperlinks import ExtractHyperlinksTool
from langchain.tools.playwright.extract_text import ExtractTextTool
from langchain.tools.playwright.get_elements import GetElementsTool
from langchain.tools.playwright.navigate import NavigateTool
from langchain.tools.playwright.navigate_back import NavigateBackTool
from langchain.tools.playwright.utils import create_async_playwright_browser

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.sync_api import Browser as SyncBrowser


class PlayWrightBrowserToolkit(BaseToolkit):
    """Toolkit for web browser tools."""

    sync_browser: SyncBrowser = Field(default_factory=create_async_playwright_browser)
    async_browser: AsyncBrowser = Field(default_factory=create_async_playwright_browser)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator
    def check_args(cls, values: dict) -> dict:
        """Check that the arguments are valid."""
        try:
            from playwright.async_api import Browser as AsyncBrowser  # noqa: F401
            from playwright.sync_api import Browser as SyncBrowser  # noqa: F401
        except ImportError:
            raise ValueError(
                "The 'playwright' package is required to use this tool."
                " Please install it with 'pip install playwright'."
            )
        if values.get("async_browser") is None and values.get("sync_browser") is None:
            raise ValueError("Either async_browser or sync_browser must be specified.")
        return values

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tool_classes: List[Type[BaseBrowserToolMixin]] = [
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
        from playwright.async_api import Browser as AsyncBrowser
        from playwright.sync_api import Browser as SyncBrowser

        cls.update_forward_refs(AsyncBrowser=AsyncBrowser)
        cls.update_forward_refs(SyncBrowser=SyncBrowser)
        return cls(sync_browser=sync_browser, async_browser=async_browser)
