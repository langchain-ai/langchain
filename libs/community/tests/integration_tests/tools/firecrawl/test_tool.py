import pytest
from unittest.mock import patch, MagicMock
from langchain_community.tools.firecrawl.tool import FirecrawlScrapeWebsiteTool
from firecrawl import FirecrawlApp

@pytest.fixture(scope="module")
def firecrawl_tool():
    """Fixture to create a FirecrawlScrapeWebsiteTool instance."""
    with patch.object(FirecrawlApp, 'map_url') as mock_map_url, \
         patch.object(FirecrawlApp, 'scrape_url') as mock_scrape_url:
        
        mock_map_url.return_value = {"links": ["https://example.com/page1", "https://example.com/page2"]}
        mock_scrape_url.return_value = {"markdown": "Sample markdown content"}
        
        tool = FirecrawlScrapeWebsiteTool(api_key="fake_api_key")
        yield tool

@pytest.mark.asyncio
async def test_integration_scrape_all_urls(firecrawl_tool: FirecrawlScrapeWebsiteTool):
    result = await firecrawl_tool.scrape_all_urls("https://example.com")
    assert "https://example.com/page1" in result
    assert "Sample markdown content" in result

@pytest.mark.asyncio
async def test_integration_error_handling(firecrawl_tool: FirecrawlScrapeWebsiteTool):
    with patch.object(firecrawl_tool.app, 'map_url', side_effect=Exception("API Error")):
        with pytest.raises(RuntimeError, match="Failed to map website: https://example.com"):
            await firecrawl_tool.scrape_all_urls("https://example.com")

@pytest.mark.asyncio
async def test_integration_invoke(firecrawl_tool: FirecrawlScrapeWebsiteTool):
    result = await firecrawl_tool._arun("https://example.com")
    assert "https://example.com/page1" in result
    assert "Sample markdown content" in result

# Add this new test to specifically check async_scrape_url
@pytest.mark.asyncio
async def test_async_scrape_url(firecrawl_tool: FirecrawlScrapeWebsiteTool):
    mock_app = MagicMock(spec=FirecrawlApp)
    mock_app.scrape_url.return_value = {"markdown": "Test content"}
    
    result = await firecrawl_tool.async_scrape_url(mock_app, "https://example.com/test")
    assert result == "Test content"

pytestmark = pytest.mark.asyncio