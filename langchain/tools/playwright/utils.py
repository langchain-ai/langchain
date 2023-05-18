"""Utilities for the Playwright browser tools."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Coroutine, TypeVar

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.async_api import Page as AsyncPage
    from playwright.sync_api import Browser as SyncBrowser
    from playwright.sync_api import Page as SyncPage


async def aget_current_page(browser: AsyncBrowser) -> AsyncPage:
    if not browser.contexts:
        context = await browser.new_context()
        return await context.new_page()
    context = browser.contexts[0]  # Assuming you're using the default browser context
    if not context.pages:
        return await context.new_page()
    # Assuming the last page in the list is the active one
    return context.pages[-1]


def get_current_page(browser: SyncBrowser) -> SyncPage:
    if not browser.contexts:
        context = browser.new_context()
        return context.new_page()
    context = browser.contexts[0]  # Assuming you're using the default browser context
    if not context.pages:
        return context.new_page()
    # Assuming the last page in the list is the active one
    return context.pages[-1]


def create_async_playwright_browser(headless: bool = True) -> AsyncBrowser:
    from playwright.async_api import async_playwright

    browser = run_async(async_playwright().start())
    return run_async(browser.chromium.launch(headless=headless))


def create_sync_playwright_browser(headless: bool = True) -> SyncBrowser:
    from playwright.sync_api import sync_playwright

    browser = sync_playwright().start()
    return browser.chromium.launch(headless=headless)


T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)
