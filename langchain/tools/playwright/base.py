from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, root_validator

from langchain.tools.base import BaseTool
from langchain.tools.playwright.utils import create_playwright_browser, run_async

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser


class BaseBrowserTool(BaseTool):
    """Base class for browser tools."""

    browser: AsyncBrowser = Field(default_factory=create_playwright_browser)

    @root_validator
    def check_args(cls, values: dict) -> dict:
        """Check that the arguments are valid."""
        try:
            from playwright.async_api import Browser as AsyncBrowser  # noqa: F401
        except ImportError:
            raise ValueError(
                "The 'playwright' package is required to use this tool."
                " Please install it with 'pip install playwright'."
            )
        return values

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Use the tool."""
        return run_async(self._arun(*args, **kwargs))

    @classmethod
    def from_browser(cls, browser: AsyncBrowser) -> BaseBrowserTool:
        from playwright.async_api import Browser as AsyncBrowser

        cls.update_forward_refs(AsyncBrowser=AsyncBrowser)
        return cls(browser=browser)
