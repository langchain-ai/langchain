"""Tests for the Playwright URL loader"""

from langchain.document_loaders import PlaywrightURLLoader


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
