from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Type

from langchain.pydantic_v1 import root_validator
from langchain.tools.base import BaseTool

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


def lazy_import_playwright_browsers() -> Tuple[Type[AsyncBrowser], Type[SyncBrowser]]:
    """
    Lazy import playwright browsers.

    Returns:
        Tuple[Type[AsyncBrowser], Type[SyncBrowser]]:
            AsyncBrowser and SyncBrowser classes.
    """
    try:
        from playwright.async_api import Browser as AsyncBrowser  # noqa: F401
        from playwright.sync_api import Browser as SyncBrowser  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'playwright' package is required to use the playwright tools."
            " Please install it with 'pip install playwright'."
        )
    return AsyncBrowser, SyncBrowser


class BaseBrowserTool(BaseTool):
    """Base class for browser tools."""

    sync_browser: Optional["SyncBrowser"] = None
    async_browser: Optional["AsyncBrowser"] = None

    @root_validator
    def validate_browser_provided(cls, values: dict) -> dict:
        """Check that the arguments are valid."""
        lazy_import_playwright_browsers()
        if values.get("async_browser") is None and values.get("sync_browser") is None:
            raise ValueError("Either async_browser or sync_browser must be specified.")
        return values

    @classmethod
    def from_browser(
        cls,
        sync_browser: Optional[SyncBrowser] = None,
        async_browser: Optional[AsyncBrowser] = None,
    ) -> BaseBrowserTool:
        """Instantiate the tool."""
        lazy_import_playwright_browsers()
        return cls(sync_browser=sync_browser, async_browser=async_browser)
