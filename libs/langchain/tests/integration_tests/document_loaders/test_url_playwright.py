"""Tests for the Playwright URL loader"""
import pytest

from langchain.document_loaders import PlaywrightURLLoader


class TestEvaluator(PageEvaluator):
    """A simple evaluator for testing purposes."""

    def evaluate(self, page, browser, response):
        return "test"

    async def evaluate_async(self, page, browser, response):
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


@pytest.mark.asyncio
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
        page_evaluator=TestEvaluator(),
        continue_on_failure=False,
        headless=True,
    )
    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].page_content == "test-"


@pytest.mark.asyncio
async def test_playwright_async_url_loader_with_custom_evaluator() -> None:
    """Test Playwright async URL loader with a custom evaluator."""
    urls = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    loader = PlaywrightURLLoader(
        urls=urls,
        page_evaluator=TestEvaluator(),
        continue_on_failure=False,
        headless=True,
    )
    docs = await loader.aload()
    assert len(docs) == 2
    assert docs[0].page_content == "test"
