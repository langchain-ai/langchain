"""Tests for the Playwright URL loader"""

from typing import TYPE_CHECKING

from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_community.document_loaders.url_playwright import PlaywrightEvaluator

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.async_api import Page as AsyncPage
    from playwright.async_api import Response as AsyncResponse
    from playwright.sync_api import Browser, Page, Response


class TestEvaluator(PlaywrightEvaluator):
    """A simple evaluator for testing purposes."""

    def evaluate(self, page: "Page", browser: "Browser", response: "Response") -> str:
        return "test"

    async def evaluate_async(
        self, page: "AsyncPage", browser: "AsyncBrowser", response: "AsyncResponse"
    ) -> str:
        return "test"


def test_playwright_url_loader() -> None:
    """Test Playwright URL loader."""
    urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://goo.gl/maps/NDSHwePEyaHMFGwh8",
        "https://techmeme.com",
        "https://techcrunch.com",
    ]
    loader = PlaywrightURLLoader(
        urls=urls,
        remove_selectors=["header", "footer"],
        continue_on_failure=False,
        headless=True,
    )
    docs = loader.load()
    assert len(docs) > 0


async def test_playwright_async_url_loader() -> None:
    """Test Playwright async URL loader."""
    urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://goo.gl/maps/NDSHwePEyaHMFGwh8",
        "https://techmeme.com",
        "https://techcrunch.com",
    ]
    loader = PlaywrightURLLoader(
        urls=urls,
        remove_selectors=["header", "footer"],
        continue_on_failure=False,
        headless=True,
    )
    docs = await loader.aload()
    assert len(docs) > 0


def test_playwright_url_loader_with_custom_evaluator() -> None:
    """Test Playwright URL loader with a custom evaluator."""
    urls = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    loader = PlaywrightURLLoader(
        urls=urls,
        evaluator=TestEvaluator(),
        continue_on_failure=False,
        headless=True,
    )
    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].page_content == "test"


async def test_playwright_async_url_loader_with_custom_evaluator() -> None:
    """Test Playwright async URL loader with a custom evaluator."""
    urls = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    loader = PlaywrightURLLoader(
        urls=urls,
        evaluator=TestEvaluator(),
        continue_on_failure=False,
        headless=True,
    )
    docs = await loader.aload()
    assert len(docs) == 1
    assert docs[0].page_content == "test"
