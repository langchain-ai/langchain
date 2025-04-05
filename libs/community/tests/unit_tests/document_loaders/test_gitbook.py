from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from langchain_community.document_loaders.gitbook import GitbookLoader


@pytest.fixture
def mock_soups() -> Tuple[BeautifulSoup, BeautifulSoup]:
    # Create mock soup with loc elements for sitemap testing
    sitemap_content = """
    <urlset>
        <url><loc>https://example.com/page1</loc></url>
        <url><loc>https://example.com/page2</loc></url>
        <url><loc>https://example.com/page3</loc></url>
    </urlset>
    """
    mock_sitemap_soup = BeautifulSoup(sitemap_content, "html.parser")

    # Create mock soup for page content
    page_content = """
    <html>
        <body>
            <main>
                <h1>Test Page</h1>
                <p>This is test content.</p>
            </main>
        </body>
    </html>
    """
    mock_page_soup = BeautifulSoup(page_content, "html.parser")
    return mock_sitemap_soup, mock_page_soup


@patch("langchain_community.document_loaders.gitbook.GitbookLoader.scrape")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader.scrape_all")
def test_lazy_load_with_custom_sitemap(
    mock_scrape_all: MagicMock,
    mock_scrape: MagicMock,
    mock_soups: Tuple[BeautifulSoup, BeautifulSoup],
) -> None:
    # Setup the mocks
    mock_sitemap_soup, mock_page_soup = mock_soups
    mock_scrape.return_value = mock_sitemap_soup
    mock_scrape_all.return_value = [
        mock_page_soup,
        mock_page_soup,
        mock_page_soup,
    ]

    # Create loader with custom sitemap URL
    loader = GitbookLoader(
        web_page="https://example.com",
        load_all_paths=True,
        sitemap_url="https://example.com/sitemap-pages.xml",
    )

    # Get the documents
    docs = list(loader.lazy_load())

    # Check that we got docs for each path in the sitemap
    assert len(docs) == 3
    for doc in docs:
        assert doc.metadata["title"] == "Test Page"
        assert "This is test content." in doc.page_content


@patch("langchain_community.document_loaders.gitbook.GitbookLoader.scrape")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader.scrape_all")
def test_recursive_sitemap_handling(
    mock_scrape_all: MagicMock, mock_scrape: MagicMock
) -> None:
    """Test that GitbookLoader correctly handles recursive sitemap structures."""

    # Mock sitemap index (contains references to other sitemaps ending in -pages.xml)
    sitemap_index_content = """
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <sitemap>
            <loc>https://example.com/sitemap-pages.xml</loc>
        </sitemap>
        <sitemap>
            <loc>https://example.com/api/sitemap-pages.xml</loc>
        </sitemap>
         <sitemap>
            <loc>https://example.com/changelog/sitemap-pages.xml</loc>
        </sitemap>
    </sitemapindex>
    """
    mock_sitemap_index = BeautifulSoup(sitemap_index_content, "html.parser")

    # Mock child sitemaps with actual content URLs
    sitemap_pages_content = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/page1</loc></url>
        <url><loc>https://example.com/page2</loc></url>
    </urlset>
    """
    mock_sitemap_pages = BeautifulSoup(sitemap_pages_content, "html.parser")

    sitemap_api_pages_content = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/api/endpoint1</loc></url>
    </urlset>
    """
    mock_sitemap_api_pages = BeautifulSoup(sitemap_api_pages_content, "html.parser")

    sitemap_changelog_pages_content = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/changelog/update1</loc></url>
        <url><loc>https://example.com/changelog/update2</loc></url>
    </urlset>
    """
    mock_sitemap_changelog_pages = BeautifulSoup(
        sitemap_changelog_pages_content, "html.parser"
    )

    # Mock page content
    page_content = """
    <html>
        <body>
            <main>
                <h1>Test Page</h1>
                <p>This is test content.</p>
            </main>
        </body>
    </html>
    """
    mock_page = BeautifulSoup(page_content, "html.parser")

    # Track the order of sitemap URLs to be processed
    sitemap_urls_order = [
        "https://example.com/sitemap.xml",
        "https://example.com/sitemap-pages.xml",
        "https://example.com/api/sitemap-pages.xml",
        "https://example.com/changelog/sitemap-pages.xml",
    ]
    sitemap_index = 0

    # Configure the mock to return different responses based on current_web_paths
    def scrape_side_effect(*args: Any, **kwargs: Any) -> BeautifulSoup:
        nonlocal sitemap_index
        web_path = sitemap_urls_order[sitemap_index]
        sitemap_index = (sitemap_index + 1) % len(sitemap_urls_order)

        if web_path == "https://example.com/sitemap.xml":
            return mock_sitemap_index
        elif web_path == "https://example.com/sitemap-pages.xml":
            return mock_sitemap_pages
        elif web_path == "https://example.com/api/sitemap-pages.xml":
            return mock_sitemap_api_pages
        elif web_path == "https://example.com/changelog/sitemap-pages.xml":
            return mock_sitemap_changelog_pages
        else:
            # Assume any other URL is a content page
            return mock_page

    mock_scrape.side_effect = scrape_side_effect

    # Mock the scrape_all method to return page content for all URLs found
    # Total URLs = 2 (pages) + 1 (api) + 2 (changelog) = 5
    mock_scrape_all.return_value = [mock_page] * 5

    # Create loader with the main sitemap URL
    loader = GitbookLoader(
        web_page="https://example.com",  # base_url is derived from this
        load_all_paths=True,
        sitemap_url="https://example.com/sitemap.xml",  # Explicitly set sitemap URL
    )

    # Get the documents
    docs = list(loader.lazy_load())

    # Verify we got the expected number of documents (2 + 1 + 2 = 5)
    assert len(docs) == 5

    # Check that scrape_all was called with the correct content page URLs
    assert mock_scrape_all.call_count == 1
    urls_arg = mock_scrape_all.call_args[0][0]
    assert len(urls_arg) == 5

    # Use sets for easier comparison, order doesn't matter
    expected_content_urls = {
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/api/endpoint1",
        "https://example.com/changelog/update1",
        "https://example.com/changelog/update2",
    }
    assert set(urls_arg) == expected_content_urls

    # Verify the content of the returned documents
    for doc in docs:
        assert "Test Page" in doc.page_content
        assert "This is test content." in doc.page_content
        assert doc.metadata["title"] == "Test Page"


@patch("langchain_community.document_loaders.gitbook.GitbookLoader.scrape")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader.scrape_all")
def test_load_method(
    mock_scrape_all: MagicMock,
    mock_scrape: MagicMock,
) -> None:
    """Test the load() method which returns a list of documents."""
    # Create mock content
    mock_sitemap_content = """
    <urlset>
        <url><loc>https://example.com/page1</loc></url>
        <url><loc>https://example.com/page2</loc></url>
    </urlset>
    """
    mock_sitemap = BeautifulSoup(mock_sitemap_content, "html.parser")

    # Create mock page soup
    page_content = """
    <html>
        <body>
            <main>
                <h1>Test Page</h1>
                <p>This is test content.</p>
            </main>
        </body>
    </html>
    """
    mock_page = BeautifulSoup(page_content, "html.parser")

    # Setup mocks
    mock_scrape.return_value = mock_sitemap
    mock_scrape_all.return_value = [mock_page, mock_page]

    # Create loader
    loader = GitbookLoader(
        web_page="https://example.com",
        load_all_paths=True,
    )

    # Test the load() method (non-lazy, synchronous)
    docs = loader.load()

    # Verify results
    assert isinstance(docs, list)
    assert len(docs) == 2
    for doc in docs:
        assert "Test Page" in doc.page_content
        assert "This is test content." in doc.page_content
        assert doc.metadata["title"] == "Test Page"


@patch("langchain_community.document_loaders.gitbook.GitbookLoader.ascrape_all")
@pytest.mark.asyncio
async def test_alazy_load_single_page(mock_ascrape_all: MagicMock) -> None:
    """Test the alazy_load() method for a single page."""
    # Create mock page soup
    page_content = """
    <html>
        <body>
            <main>
                <h1>Test Single Page</h1>
                <p>This is single page test content.</p>
            </main>
            <title>Test Page Title</title>
        </body>
    </html>
    """
    mock_page = BeautifulSoup(page_content, "html.parser")

    # Setup mock to return our page
    mock_ascrape_all.return_value = [mock_page]

    # Create loader for a single page
    loader = GitbookLoader(
        web_page="https://example.com/page",
        load_all_paths=False,
    )

    # Collect documents
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    # Verify results
    assert len(docs) == 1
    assert "Test Single Page" in docs[0].page_content
    assert "This is single page test content." in docs[0].page_content
    # Check source in metadata
    assert docs[0].metadata["source"] == "https://example.com/page"
    # Title may be from h1 or title element depending on implementation
    assert "title" in docs[0].metadata
    assert docs[0].metadata["title"] in ["Test Single Page", "Test Page Title"]


@patch("langchain_community.document_loaders.gitbook.GitbookLoader.ascrape_all")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader._aprocess_sitemap")
@pytest.mark.asyncio
async def test_alazy_load_recursive_sitemap(
    mock_aprocess_sitemap: MagicMock, mock_ascrape_all: MagicMock
) -> None:
    """Test the alazy_load() method with recursive sitemap processing."""
    # Create mock sitemaps and content
    sitemap_content = """
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <sitemap><loc>https://example.com/sitemap1.xml</loc></sitemap>
    </sitemapindex>
    """
    mock_sitemap = BeautifulSoup(sitemap_content, "html.parser")

    # Create mock page soups
    page_content = """
    <html>
        <body>
            <main>
                <h1>Test Async Page</h1>
                <p>This is async test content.</p>
            </main>
        </body>
    </html>
    """
    mock_page1 = BeautifulSoup(page_content, "html.parser")
    mock_page2 = BeautifulSoup(page_content, "html.parser")
    mock_page3 = BeautifulSoup(page_content, "html.parser")

    # Setup mock returns for different calls to ascrape_all
    mock_ascrape_all.side_effect = [
        # First call: Get the initial sitemap
        [mock_sitemap],
        # Second call: Get the content pages
        [mock_page1, mock_page2, mock_page3],
    ]

    # Setup _aprocess_sitemap to return some paths
    mock_aprocess_sitemap.return_value = [
        "/async-page1",
        "/async-page2",
        "/async-page3",
    ]

    # Create loader with sitemap URL
    loader = GitbookLoader(
        web_page="https://example.com",
        load_all_paths=True,
        sitemap_url="https://example.com/sitemap.xml",
    )

    # Collect documents
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    # Verify results
    assert len(docs) == 3
    assert mock_ascrape_all.call_count == 2
    assert mock_aprocess_sitemap.call_count == 1

    # Check that the first call to ascrape_all had the expected arguments
    first_call_args, first_call_kwargs = mock_ascrape_all.call_args_list[0]
    assert first_call_args[0] == ["https://example.com/sitemap.xml"]
    # Check parser parameter in kwargs instead of positional args
    assert first_call_kwargs.get("parser") == "xml"

    # Second call should fetch content URLs
    expected_urls = [
        "https://example.com/async-page1",
        "https://example.com/async-page2",
        "https://example.com/async-page3",
    ]
    second_call_args = mock_ascrape_all.call_args_list[1][0][0]
    assert set(second_call_args) == set(expected_urls)

    # Verify document content
    for doc in docs:
        assert "Test Async Page" in doc.page_content
        assert "This is async test content." in doc.page_content
        assert doc.metadata["title"] == "Test Async Page"
