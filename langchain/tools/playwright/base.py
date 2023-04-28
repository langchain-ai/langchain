from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field, root_validator

from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,
)

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.sync_api import Browser as SyncBrowser


class BaseBrowserToolMixin(BaseModel):
    """Base class for browser tools."""

    async_browser: Optional[AsyncBrowser] = Field(
        default_factory=create_async_playwright_browser
    )
    sync_browser: Optional[SyncBrowser] = Field(
        default_factory=create_sync_playwright_browser
    )

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

    @classmethod
    def from_browser(
        cls,
        sync_browser: Optional[SyncBrowser] = None,
        async_browser: Optional[AsyncBrowser] = None,
    ) -> BaseBrowserToolMixin:
        from playwright.async_api import Browser as AsyncBrowser
        from playwright.sync_api import Browser as SyncBrowser

        cls.update_forward_refs(AsyncBrowser=AsyncBrowser)
        cls.update_forward_refs(SyncBrowser=SyncBrowser)
        return cls(sync_browser=sync_browser, async_browser=async_browser)
