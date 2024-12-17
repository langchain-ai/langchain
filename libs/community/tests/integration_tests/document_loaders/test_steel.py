"""Tests for the Steel Web Loader."""
import os
import pytest
from typing import Generator
from unittest.mock import MagicMock, AsyncMock, patch
from playwright.async_api import Browser, BrowserContext, Page, Playwright

from langchain_community.document_loaders import SteelWebLoader

requires_api_key = pytest.mark.skipif(
    not os.getenv("STEEL_API_KEY"),
    reason="Test requires STEEL_API_KEY environment variable"
)

@pytest.fixture
def mock_page() -> Generator[MagicMock, None, None]:
    """Create a mock Playwright page."""
    page = MagicMock(spec=Page)
    page.goto = AsyncMock()
    page.inner_text = AsyncMock(return_value="Sample page content")
    page.content = AsyncMock(return_value="<html><body>Sample page content</body></html>")
    yield page

@pytest.fixture
def mock_context(mock_page) -> Generator[MagicMock, None, None]:
    """Create a mock browser context."""
    context = MagicMock(spec=BrowserContext)
    context.new_page = AsyncMock(return_value=mock_page)
    yield context

@pytest.fixture
def mock_browser(mock_context) -> Generator[MagicMock, None, None]:
    """Create a mock browser."""
    browser = MagicMock(spec=Browser)
    browser.contexts = [mock_context]
    browser.close = AsyncMock()
    yield browser

@pytest.fixture
def mock_playwright(mock_browser) -> Generator[MagicMock, None, None]:
    """Create a mock Playwright instance."""
    playwright = MagicMock(spec=Playwright)
    playwright.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
    playwright.stop = AsyncMock()
    yield playwright

def test_init_no_api_key():
    """Test initialization without API key."""
    with pytest.raises(ValueError, match="steel_api_key must be provided"):
        SteelWebLoader("https://example.com")

def test_init_invalid_strategy():
    """Test initialization with invalid extraction strategy."""
    with pytest.raises(ValueError, match="Invalid extract_strategy"):
        SteelWebLoader(
            "https://example.com",
            steel_api_key="test-key",
            extract_strategy="invalid"
        )

@patch("langchain_community.document_loaders.steel.async_playwright")
async def test_load_text_content(
    mock_async_playwright,
    mock_playwright,
    mock_page
):
    """Test loading page with text extraction strategy."""
    mock_async_playwright.return_value.start = AsyncMock(return_value=mock_playwright)
    
    loader = SteelWebLoader(
        "https://example.com",
        steel_api_key="test-key"
    )
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].page_content == "Sample page content"
    assert docs[0].metadata["source"] == "https://example.com"
    assert docs[0].metadata["steel_session_id"]
    assert docs[0].metadata["steel_session_viewer_url"]
    assert docs[0].metadata["extract_strategy"] == "text"
    
    # Verify page navigation
    mock_page.goto.assert_awaited_once_with(
        "https://example.com",
        wait_until="networkidle",
        timeout=30000
    )

@patch("langchain_community.document_loaders.steel.async_playwright")
async def test_load_html_content(
    mock_async_playwright,
    mock_playwright,
    mock_page
):
    """Test loading page with HTML extraction strategy."""
    mock_async_playwright.return_value.start = AsyncMock(return_value=mock_playwright)
    
    loader = SteelWebLoader(
        "https://example.com",
        steel_api_key="test-key",
        extract_strategy="html"
    )
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].page_content == "<html><body>Sample page content</body></html>"
    assert docs[0].metadata["extract_strategy"] == "html"

@patch("langchain_community.document_loaders.steel.async_playwright")
async def test_load_with_error(mock_async_playwright):
    """Test error handling during page load."""
    mock_async_playwright.return_value.start = AsyncMock(
        side_effect=Exception("Connection failed")
    )
    
    loader = SteelWebLoader(
        "https://example.com",
        steel_api_key="test-key"
    )
    
    with pytest.raises(Exception, match="Connection failed"):
        loader.load()

@requires_api_key
def test_integration_basic():
    """Integration test for basic functionality."""
    loader = SteelWebLoader(
        "https://example.com",
        extract_strategy="text"
    )
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].page_content
    assert docs[0].metadata["source"] == "https://example.com"
    assert docs[0].metadata["steel_session_id"]
    assert docs[0].metadata["steel_session_viewer_url"]

@requires_api_key
@pytest.mark.parametrize("strategy", ["text", "markdown", "html"])
def test_integration_strategies(strategy):
    """Integration test for different extraction strategies."""
    loader = SteelWebLoader(
        "https://example.com",
        extract_strategy=strategy
    )
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].metadata["extract_strategy"] == strategy
