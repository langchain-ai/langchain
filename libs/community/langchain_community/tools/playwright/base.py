from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, Type

from langchain_core.tools import BaseTool
from langchain_core.utils import guard_import
from pydantic import model_validator

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
    return (
        guard_import(module_name="playwright.async_api").Browser,
        guard_import(module_name="playwright.sync_api").Browser,
    )


class BaseBrowserTool(BaseTool):
    """Base class for browser tools."""

    sync_browser: Optional["SyncBrowser"] = None
    async_browser: Optional["AsyncBrowser"] = None

    @model_validator(mode="before")
    @classmethod
    def validate_browser_provided(cls, values: dict) -> Any:
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
        return cls(sync_browser=sync_browser, async_browser=async_browser)  # type: ignore[call-arg]
