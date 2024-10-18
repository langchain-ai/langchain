import pytest
from unittest.mock import patch
from app import FirecrawlScrapeWebsiteTool
import asyncio

@pytest.fixture
def mock_firecrawl_app():
    with patch('app.FirecrawlApp') as MockFirecrawlApp:
        mock_app = MockFirecrawlApp.return_value
        mock_app.map_url.return_value = {'links': ['https://example.com/page1', 'https://example.com/page2']}
        mock_app.scrape_url.return_value = {'markdown': 'Sample markdown content'}
        yield mock_app

def test_scrape_all_urls(mock_firecrawl_app):
    tool = FirecrawlScrapeWebsiteTool(api_key='fake_api_key')
    result = asyncio.run(tool.scrape_all_urls('https://example.com'))
    assert 'https://example.com/page1' in result
    assert 'Sample markdown content' in result

def test_map_website(mock_firecrawl_app):
    tool = FirecrawlScrapeWebsiteTool(api_key='fake_api_key')
    urls = tool.map_website(mock_firecrawl_app, 'https://example.com')
    assert urls == ['https://example.com/page1', 'https://example.com/page2']

def test_async_scrape_url(mock_firecrawl_app):
    tool = FirecrawlScrapeWebsiteTool(api_key='fake_api_key')
    content = asyncio.run(tool.async_scrape_url(mock_firecrawl_app, 'https://example.com/page1'))
    assert content == 'Sample markdown content'

def test_error_handling_in_map_website():
    with patch('app.FirecrawlApp') as MockFirecrawlApp:
        mock_app = MockFirecrawlApp.return_value
        mock_app.map_url.side_effect = Exception("Test exception")
        tool = FirecrawlScrapeWebsiteTool(api_key='fake_api_key')
        with pytest.raises(RuntimeError, match="Failed to map website: https://example.com"):
            tool.map_website(mock_app, 'https://example.com')

def test_error_handling_in_async_scrape_url():
    with patch('app.FirecrawlApp') as MockFirecrawlApp:
        mock_app = MockFirecrawlApp.return_value
        mock_app.scrape_url.side_effect = Exception("Test exception")
        tool = FirecrawlScrapeWebsiteTool(api_key='fake_api_key')
        with pytest.raises(RuntimeError, match="Failed to scrape URL: https://example.com/page1"):
            asyncio.run(tool.async_scrape_url(mock_app, 'https://example.com/page1'))
