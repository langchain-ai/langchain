"""Playwright web browser toolkit."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Type, cast

from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.tools import BaseTool, BaseToolkit

from langchain_community.tools.playwright.base import (
    BaseBrowserTool,
    lazy_import_playwright_browsers,
)
from langchain_community.tools.playwright.click import ClickTool
from langchain_community.tools.playwright.current_page import CurrentWebPageTool
from langchain_community.tools.playwright.extract_hyperlinks import (
    ExtractHyperlinksTool,
)
from langchain_community.tools.playwright.extract_text import ExtractTextTool
from langchain_community.tools.playwright.get_elements import GetElementsTool
from langchain_community.tools.playwright.navigate import NavigateTool
from langchain_community.tools.playwright.navigate_back import NavigateBackTool

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
    """Toolkit for PlayWright browser tools.

    **Security Note**: This toolkit provides code to control a web-browser.

        Careful if exposing this toolkit to end-users. The tools in the toolkit
        are capable of navigating to arbitrary webpages, clicking on arbitrary
        elements, and extracting arbitrary text and hyperlinks from webpages.

        Specifically, by default this toolkit allows navigating to:

        - Any URL (including any internal network URLs)
        - And local files

        If exposing to end-users, consider limiting network access to the
        server that hosts the agent; in addition, consider it is advised
        to create a custom NavigationTool wht an args_schema that limits the URLs
        that can be navigated to (e.g., only allow navigating to URLs that
        start with a particular prefix).

        Remember to scope permissions to the minimal permissions necessary for
        the application. If the default tool selection is not appropriate for
        the application, consider creating a custom toolkit with the appropriate
        tools.

        See https://python.langchain.com/docs/security for more information.

    Parameters:
        sync_browser: Optional. The sync browser. Default is None.
        async_browser: Optional. The async browser. Default is None.
    """

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
        """Instantiate the toolkit.

        Args:
            sync_browser: Optional. The sync browser. Default is None.
            async_browser: Optional. The async browser. Default is None.

        Returns:
            The toolkit.
        """
        # This is to raise a better error than the forward ref ones Pydantic would have
        lazy_import_playwright_browsers()
        return cls(sync_browser=sync_browser, async_browser=async_browser)
