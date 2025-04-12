from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from langchain_community.document_loaders.gitbook import GitbookLoader

# These tests mock the XML parsing functionality to avoid requiring lxml.
# The underlying GitbookLoader class may use lxml for XML parsing when available,
# but the tests have been structured to work regardless of whether lxml is installed.


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


@patch("langchain_community.document_loaders.gitbook.GitbookLoader._create_web_loader")
def test_lazy_load_with_custom_sitemap(
    mock_create_loader: MagicMock,
    mock_soups: Tuple[BeautifulSoup, BeautifulSoup],
) -> None:
    """Test loading with a custom sitemap URL."""
    # Setup the mocks
    mock_sitemap_soup, mock_page_soup = mock_soups

    mock_web_loader = MagicMock()
    mock_create_loader.return_value = mock_web_loader
    mock_web_loader.scrape.return_value = mock_sitemap_soup
    mock_web_loader.scrape_all.return_value = [
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


@patch("langchain_community.document_loaders.gitbook.GitbookLoader._create_web_loader")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader._process_sitemap")
def test_recursive_sitemap_handling(
    mock_process_sitemap: MagicMock, mock_create_loader: MagicMock
) -> None:
    """Test that GitbookLoader correctly handles recursive sitemap structures."""
    # Mock the _process_sitemap to return all paths from sitemap
    mock_process_sitemap.return_value = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/api/endpoint1",
        "https://example.com/changelog/update1",
        "https://example.com/changelog/update2",
    ]

    # Mock the web loader
    mock_web_loader = MagicMock()
    mock_create_loader.return_value = mock_web_loader

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

    # Setup the scrape return value (just needs to be something)
    mock_web_loader.scrape.return_value = "Mock sitemap content"

    # Mock the scrape_all method to return page content for all URLs found
    mock_web_loader.scrape_all.return_value = [mock_page] * 5

    # Create loader with the main sitemap URL
    loader = GitbookLoader(
        web_page="https://example.com",  # base_url is derived from this
        load_all_paths=True,
        sitemap_url="https://example.com/sitemap.xml",  # Explicitly set sitemap URL
    )

    # Get the documents
    docs = list(loader.lazy_load())

    # Verify we got the expected number of documents
    assert len(docs) == 5

    # Check that scrape_all was called with the correct content page URLs
    assert mock_web_loader.scrape_all.call_count == 1
    urls_arg = mock_web_loader.scrape_all.call_args[0][0]
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


@patch("langchain_community.document_loaders.gitbook.GitbookLoader._create_web_loader")
def test_load_method(mock_create_loader: MagicMock) -> None:
    """Test the load() method which returns a list of documents."""
    # Mock the web loader
    mock_web_loader = MagicMock()
    mock_create_loader.return_value = mock_web_loader

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
    mock_web_loader.scrape.return_value = mock_sitemap
    mock_web_loader.scrape_all.return_value = [mock_page, mock_page]

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


@patch("langchain_community.document_loaders.web_base.WebBaseLoader.ascrape_all")
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


@patch("langchain_community.document_loaders.web_base.WebBaseLoader.ascrape_all")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader._aprocess_sitemap")
@pytest.mark.asyncio
async def test_alazy_load_recursive_sitemap(
    mock_aprocess_sitemap: MagicMock, mock_ascrape_all: MagicMock
) -> None:
    """Test the alazy_load() method with recursive sitemap processing."""
    # Setup mock for first call to get sitemap - content doesn't matter
    # as we're mocking _aprocess_sitemap which is called after this
    mock_sitemap = "This is a sitemap, but content doesn't matter"

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
        "https://example.com/async-page1",
        "https://example.com/async-page2",
        "https://example.com/async-page3",
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
    # XML parsing is now handled by _aprocess_sitemap

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


def test_ssrf_protection_validation() -> None:
    """Test SSRF protection by validating URLs against allowed domains."""

    # Test 1: Default behavior - should only allow the domain from web_page
    loader = GitbookLoader(web_page="https://example.com/docs")

    # Verify allowed_domains was set to include only example.com
    assert loader.allowed_domains == {"example.com"}

    # Check safe URL (same domain)
    assert loader._is_url_allowed("https://example.com/path/page.html") is True

    # Check localhost URL - should be rejected
    assert loader._is_url_allowed("http://localhost:8000/foo") is False
    assert loader._is_url_allowed("https://localhost:8000/foo") is False
    assert loader._is_url_allowed("http://127.0.0.1/admin") is False

    # Test filtering a list of URLs
    test_url_list: List[str] = []
    loader._safe_add_url(test_url_list, "https://example.com/good", "test")
    loader._safe_add_url(test_url_list, "https://localhost:8000/foo", "test")
    loader._safe_add_url(test_url_list, "https://evil.com/hack", "test")

    # Only the allowed domain URL should be in the list
    assert len(test_url_list) == 1
    assert test_url_list[0] == "https://example.com/good"

    # Test 2: Explicit allowed_domains
    loader = GitbookLoader(
        web_page="https://docs.example.org/start",
        allowed_domains={"docs.example.org", "api.example.org"},
    )

    # Check URLs against the explicitly allowed domains
    assert loader._is_url_allowed("https://docs.example.org/page") is True
    assert loader._is_url_allowed("https://api.example.org/v2/endpoint") is True
    assert loader._is_url_allowed("https://example.org/outside") is False
    assert loader._is_url_allowed("https://localhost:8000/foo") is False

    # Test 3: Try to create loader with disallowed initial URL - should raise ValueError
    with pytest.raises(ValueError):
        GitbookLoader(
            web_page="https://attacker.com/page", allowed_domains={"example.com"}
        )


@patch("langchain_community.document_loaders.gitbook.GitbookLoader._process_sitemap")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader._create_web_loader")
def test_url_filtering_during_load(
    mock_create_loader: MagicMock, mock_process_sitemap: MagicMock
) -> None:
    """Test that URLs from non-allowed domains are filtered during loading.

    This test covers the URL filtering functionality without requiring lxml,
    by mocking the XML parsing parts. This ensures the SSRF protection works
    correctly regardless of whether lxml is available.

    It tests multiple security patterns:
    1. External domain filtering
    2. Localhost access prevention
    3. Internal IP address blocking
    """
    # Setup mock WebBaseLoader
    mock_web_loader = MagicMock()
    mock_create_loader.return_value = mock_web_loader

    # Mock the _process_sitemap to directly return paths
    # This avoids needing to parse XML with lxml
    mock_process_sitemap.return_value = [
        "https://example.com/page1",  # Will be allowed - normal page
        "https://example.com/page2",  # Will be allowed - normal page
    ]

    # Set up the scrape mock to return anything
    mock_web_loader.scrape.return_value = "Mocked content (doesn't matter)"

    # Mock content pages
    def mock_scrape_all_impl(urls: List[str], **kwargs: Any) -> List[BeautifulSoup]:
        # Return dummy content for each URL
        results = []
        for url in urls:
            content = f"<main><h1>Title for {url}</h1><p>Content for {url}</p></main>"
            results.append(BeautifulSoup(content, "html.parser"))
        return results

    mock_web_loader.scrape_all.side_effect = mock_scrape_all_impl

    # Part 1: Test the normal flow with allowed URLs
    loader = GitbookLoader(
        web_page="https://example.com/docs",
        load_all_paths=True,
        sitemap_url="https://example.com/sitemap.xml",
    )

    # Patch the _safe_add_url method to verify it's called correctly
    with patch.object(loader, "_safe_add_url") as mock_safe_add:
        # Make it pass the URL through to simulate filtering
        mock_safe_add.side_effect = (
            lambda url_list, url, url_type: url_list.append(url)
            if "example.com" in url
            else None
        )

        # Collect all documents - we need to verify args after this runs
        _ = list(loader.lazy_load())

        # Verify _safe_add_url was called for each path
        assert mock_safe_add.call_count == 2

        # Verify the URLs that were requested to be added
        expected_calls: List[Tuple[Tuple[List[str], str, str]]] = [
            (([], "https://example.com/page1", "content"),),
            (([], "https://example.com/page2", "content"),),
        ]

        # We don't care about the exact list object, just the URL and type
        for i, call in enumerate(mock_safe_add.call_args_list):
            args = call[0]
            assert args[1] == expected_calls[i][0][1]  # Check URL
            assert args[2] == expected_calls[i][0][2]  # Check type

    # Part 2: Test the sitemap with mixed URLs (both allowed and disallowed)
    # Including attempts to access internal networks, localhost, etc.
    mock_process_sitemap.reset_mock()
    mock_create_loader.reset_mock()
    mock_web_loader.reset_mock()

    # This time, simulate finding both allowed and forbidden URLs in the sitemap
    mock_process_sitemap.return_value = [
        "https://example.com/page1",  # Allowed - normal page
        "https://example.com/api/internal",  # Allowed - same domain
        "https://example.com/admin",  # Allowed - same domain
        "https://malicious.com/hack",  # Disallowed - different domain
        "http://localhost:8080/admin",  # Disallowed - localhost
        "http://127.0.0.1/config",  # Disallowed - localhost IP
        "https://192.168.1.1/router",  # Disallowed - internal IP
    ]

    # Re-setup the mocks
    mock_web_loader = MagicMock()
    mock_create_loader.return_value = mock_web_loader
    mock_web_loader.scrape.return_value = "Mocked sitemap content"
    mock_web_loader.scrape_all.side_effect = mock_scrape_all_impl

    # Create a new loader with restricted allowed domains
    loader = GitbookLoader(
        web_page="https://example.com/docs",
        load_all_paths=True,
        sitemap_url="https://example.com/sitemap.xml",
    )

    # In this case, use the real _safe_add_url method to test actual filtering
    # We don't need the actual docs, just checking the filtering
    _ = list(loader.lazy_load())

    # Only the 3 allowed URLs should have been processed
    # (the 4 disallowed ones filtered out)
    assert mock_web_loader.scrape_all.call_count == 1

    # Get the URLs that were requested in scrape_all
    urls_requested = mock_web_loader.scrape_all.call_args[0][0]

    # Only example.com URLs should be in the list, and only 3 of them
    assert len(urls_requested) == 3
    assert all("example.com" in url for url in urls_requested)

    # Check that the disallowed URLs are not in the list
    assert "malicious.com" not in str(urls_requested)
    assert "localhost" not in str(urls_requested)
    assert "127.0.0.1" not in str(urls_requested)
    assert "192.168.1.1" not in str(urls_requested)
