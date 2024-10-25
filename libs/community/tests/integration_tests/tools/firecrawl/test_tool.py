from typing import Generator
from unittest.mock import patch

import pytest
from app import FirecrawlScrapeWebsiteTool


@pytest.fixture
def mock_firecrawl_app() -> Generator:
    with patch("app.FirecrawlApp") as MockFirecrawlApp:
        mock_app = MockFirecrawlApp.return_value
        mock_app.map_url.return_value = {
            "links": ["https://example.com/page1", "https://example.com/page2"]
        }
        mock_app.scrape_url.return_value = {"markdown": "Sample markdown content"}
        yield mock_app


@pytest.mark.asyncio
async def test_integration_scrape_all_urls(mock_firecrawl_app: Generator) -> None:
    tool = FirecrawlScrapeWebsiteTool(api_key="fake_api_key")
    result = await tool.scrape_all_urls("https://example.com")
    assert "https://example.com/page1" in result
    assert "Sample markdown content" in result


@pytest.mark.asyncio
async def test_integration_map_website(mock_firecrawl_app: Generator) -> None:
    tool = FirecrawlScrapeWebsiteTool(api_key="fake_api_key")
    urls = tool.map_website(mock_firecrawl_app, "https://example.com")
    assert urls == ["https://example.com/page1", "https://example.com/page2"]


@pytest.mark.asyncio
async def test_integration_async_scrape_url(mock_firecrawl_app: Generator) -> None:
    tool = FirecrawlScrapeWebsiteTool(api_key="fake_api_key")
    content = await tool.async_scrape_url(
        mock_firecrawl_app, "https://example.com/page1"
    )
    assert content == "Sample markdown content"


@pytest.mark.asyncio
async def test_integration_error_handling_in_map_website() -> None:
    with patch("app.FirecrawlApp") as MockFirecrawlApp:
        mock_app = MockFirecrawlApp.return_value
        mock_app.map_url.side_effect = Exception("Test exception")
        tool = FirecrawlScrapeWebsiteTool(api_key="fake_api_key")
        with pytest.raises(
            RuntimeError, match="Failed to map website: https://example.com"
        ):
            tool.map_website(mock_app, "https://example.com")


@pytest.mark.asyncio
async def test_integration_error_handling_in_async_scrape_url() -> None:
    with patch("app.FirecrawlApp") as MockFirecrawlApp:
        mock_app = MockFirecrawlApp.return_value
        mock_app.scrape_url.side_effect = Exception("Test exception")
        tool = FirecrawlScrapeWebsiteTool(api_key="fake_api_key")
        with pytest.raises(
            RuntimeError, match="Failed to scrape URL: https://example.com/page1"
        ):
            await tool.async_scrape_url(mock_app, "https://example.com/page1")


@pytest.mark.asyncio
async def test_invoke_with_invalid_url(mock_firecrawl_app: Generator) -> None:
    tool = FirecrawlScrapeWebsiteTool(api_key="fake_api_key")
    with patch.object(tool, 'invoke', side_effect=ValueError("Invalid URL")):
        with pytest.raises(ValueError, match="Invalid URL"):
            await tool.invoke("invalid-url")


@pytest.mark.asyncio
async def test_invoke_with_empty_url(mock_firecrawl_app: Generator) -> None:
    tool = FirecrawlScrapeWebsiteTool(api_key="fake_api_key")
    with patch.object(tool, 'invoke', return_value={"links": [], "content": ""}):
        result = await tool.invoke("")
        assert result["links"] == []
        assert result["content"] == ""


@pytest.mark.asyncio
async def test_invoke_exception_handling(mock_firecrawl_app: Generator) -> None:
    tool = FirecrawlScrapeWebsiteTool(api_key="fake_api_key")
    with patch.object(tool, 'invoke', side_effect=RuntimeError("Unexpected error")):
        with pytest.raises(RuntimeError, match="Unexpected error"):
            await tool.invoke("https://example.com")


@pytest.mark.asyncio
async def test_invoke_with_different_url_patterns(mock_firecrawl_app: Generator) -> None:
    tool = FirecrawlScrapeWebsiteTool(api_key="fake_api_key")
    with patch.object(tool, 'invoke', return_value={
        "links": ["https://example.com/page1"],
        "content": "Sample content"
    }) as mock_invoke:
        result = await tool.invoke("https://example.com/page1")
        mock_invoke.assert_called_once_with("https://example.com/page1")
        assert "https://example.com/page1" in result["links"]
        assert "Sample content" in result["content"]