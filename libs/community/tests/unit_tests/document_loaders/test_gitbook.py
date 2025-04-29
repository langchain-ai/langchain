from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from langchain_community.document_loaders.gitbook import GitbookLoader

# Note: Some tests require lxml for XML parsing mocks.


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


@pytest.mark.requires("lxml")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader._create_web_loader")
def test_lazy_load_with_custom_sitemap(
    mock_create_loader: MagicMock,
    mock_soups: Tuple[BeautifulSoup, BeautifulSoup],
) -> None:
    """Test loading with a custom sitemap URL."""
    pytest.importorskip("lxml")
    # Setup the mocks
    mock_sitemap_soup, mock_page_soup = mock_soups

    mock_web_loader = MagicMock()
    mock_create_loader.return_value = mock_web_loader
    # Simulate initial scrape returning the sitemap soup
    mock_web_loader.scrape.return_value = BeautifulSoup(
        str(mock_sitemap_soup), "lxml-xml"
    )
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
    with patch.object(loader, "_process_sitemap") as mock_process_sitemap:
        # Mock _process_sitemap to return the expected paths directly
        mock_process_sitemap.return_value = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
        ]
        docs = list(loader.lazy_load())

    # Check that we got docs for each path in the sitemap
    assert len(docs) == 3
    for doc in docs:
        assert doc.metadata["title"] == "Test Page"
        assert "This is test content." in doc.page_content


@pytest.mark.requires("lxml")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader._create_web_loader")
def test_recursive_sitemap_handling(
    mock_create_loader: MagicMock,
) -> None:
    """Test recursive sitemap handling with simplified mocks."""
    pytest.importorskip("lxml")

    # Mock web loader instance
    mock_web_loader = MagicMock()
    mock_create_loader.return_value = mock_web_loader

    # Mock initial sitemap fetch
    mock_web_loader.scrape.return_value = BeautifulSoup(
        "<sitemapindex></sitemapindex>", "lxml-xml"
    )

    # Mock content page fetch
    page_content = (
        "<html><body><main><h1>Test Page</h1><p>Content</p></main></body></html>"
    )
    mock_page = BeautifulSoup(page_content, "html.parser")
    mock_web_loader.scrape_all.return_value = [mock_page] * 5  # Assume 5 final pages

    # The key is to mock _process_sitemap directly to return the expected final URLs
    expected_final_urls = {
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/api/endpoint1",
        "https://example.com/changelog/update1",
        "https://example.com/changelog/update2",
    }

    loader = GitbookLoader(
        web_page="https://example.com",
        load_all_paths=True,
        sitemap_url="https://example.com/sitemap.xml",
    )

    # Patch _process_sitemap to return the final URLs directly
    with patch.object(
        loader, "_process_sitemap", return_value=list(expected_final_urls)
    ) as mock_proc_sitemap:
        docs = list(loader.lazy_load())

        # Verify _process_sitemap was called once with the initial soup
        mock_proc_sitemap.assert_called_once()

        # Verify scrape_all was called with the correct final URLs
        mock_web_loader.scrape_all.assert_called_once()
        urls_arg = mock_web_loader.scrape_all.call_args[0][0]
        assert set(urls_arg) == expected_final_urls

        # Verify document count
        assert len(docs) == 5


@pytest.mark.requires("lxml")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader._create_web_loader")
def test_load_method(mock_create_loader: MagicMock) -> None:
    """Test the load() method which returns a list of documents."""
    pytest.importorskip("lxml")
    # Mock the web loader
    mock_web_loader = MagicMock()
    mock_create_loader.return_value = mock_web_loader

    # Create mock sitemap
    mock_sitemap_content = """<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/page1</loc></url>
        <url><loc>https://example.com/page2</loc></url>
    </urlset>"""
    mock_sitemap = BeautifulSoup(mock_sitemap_content, "lxml-xml")

    # Create mock page soup
    page_content = (
        "<html><body><main><h1>Test Page</h1>"
        "<p>This is test content.</p></main></body></html>"
    )
    mock_page = BeautifulSoup(page_content, "html.parser")

    # Setup mocks
    mock_web_loader.scrape.return_value = mock_sitemap  # For initial sitemap load
    mock_web_loader.scrape_all.return_value = [
        mock_page,
        mock_page,
    ]  # For content pages

    # Create loader
    loader = GitbookLoader(
        web_page="https://example.com",
        load_all_paths=True,
    )

    # Patch _process_sitemap to return the final URLs
    with patch.object(
        loader,
        "_process_sitemap",
        return_value=["https://example.com/page1", "https://example.com/page2"],
    ):
        # Test the load() method
        docs = loader.load()

    # Verify results
    assert isinstance(docs, list)
    assert len(docs) == 2
    # Add other assertions


@patch("langchain_community.document_loaders.web_base.WebBaseLoader.ascrape_all")
@pytest.mark.asyncio
async def test_alazy_load_single_page(mock_ascrape_all: MagicMock) -> None:
    """Test the alazy_load() method for a single page."""
    # Create mock page soup
    page_content = (
        "<html><body><main><h1>Test Single Page</h1>"
        "<p>Single page content.</p></main><title>Test Title</title></body></html>"
    )
    mock_page = BeautifulSoup(page_content, "html.parser")
    mock_ascrape_all.return_value = [mock_page]

    # Create loader
    loader = GitbookLoader("https://example.com/page", load_all_paths=False)

    # Collect documents
    docs = [doc async for doc in loader.alazy_load()]

    # Verify results
    assert len(docs) == 1
    # Add assertions


@pytest.mark.requires("lxml")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader._create_web_loader")
@pytest.mark.asyncio
async def test_alazy_load_recursive_sitemap(mock_create_loader: MagicMock) -> None:
    """Test the alazy_load() method with recursive sitemap processing."""
    pytest.importorskip("lxml")
    # Mock web loader
    mock_web_loader = MagicMock()
    mock_create_loader.return_value = mock_web_loader

    # Mock initial sitemap fetch
    mock_sitemap = BeautifulSoup("<sitemapindex></sitemapindex>", "lxml-xml")

    # Configure ascrape_all mock for the initial sitemap fetch
    async def ascrape_all_side_effect(
        urls: List[str], **kwargs: Any
    ) -> List[BeautifulSoup]:
        if urls == ["https://example.com/sitemap.xml"]:
            assert kwargs.get("parser") == "lxml-xml"
            return [mock_sitemap]
        elif set(urls) == {
            "https://example.com/async-page1",
            "https://example.com/async-page2",
            "https://example.com/async-page3",
        }:
            # Mock content page fetch
            page_content = (
                "<html><body><main><h1>Async Page</h1>"
                "<p>Async content.</p></main></body></html>"
            )
            mock_page = BeautifulSoup(page_content, "html.parser")
            return [mock_page] * len(urls)
        else:
            raise ValueError(f"Unexpected ascrape_all call: {urls}")

    mock_web_loader.ascrape_all.side_effect = ascrape_all_side_effect

    # Create loader
    loader = GitbookLoader(
        web_page="https://example.com",
        load_all_paths=True,
        sitemap_url="https://example.com/sitemap.xml",
    )

    # Patch the _aprocess_sitemap method directly
    final_urls = [
        "https://example.com/async-page1",
        "https://example.com/async-page2",
        "https://example.com/async-page3",
    ]
    with patch.object(
        loader, "_aprocess_sitemap", return_value=final_urls
    ) as mock_aproc_sitemap:
        # Collect documents
        docs = [doc async for doc in loader.alazy_load()]

        # Verify results
        assert len(docs) == 3
        mock_aproc_sitemap.assert_awaited_once()
        # Check that ascrape_all was called at least once
        mock_web_loader.ascrape_all.assert_called()
        # Check calls to ascrape_all made by the mocked loader
        assert mock_web_loader.ascrape_all.call_count == 2  # Initial + Content
        # Check content call arguments
        content_call_args = mock_web_loader.ascrape_all.call_args_list[1][0][0]
        assert set(content_call_args) == set(final_urls)
        # Add assertions on doc content if needed


def test_ssrf_protection_validation() -> None:
    """Test SSRF protection by validating URLs against allowed domains."""
    # Test 1: Default behavior - should only allow the domain from web_page
    loader = GitbookLoader(web_page="https://example.com/docs")
    assert loader.allowed_domains == {"example.com"}
    assert loader._is_url_allowed("https://example.com/path/page.html") is True
    assert loader._is_url_allowed("ftp://example.com/file") is False
    assert loader._is_url_allowed("file:///etc/passwd") is False
    assert loader._is_url_allowed("javascript:alert(1)") is False
    assert loader._is_url_allowed("example.com/path/page.html") is False
    assert loader._is_url_allowed("http://localhost:8000/foo") is False
    assert loader._is_url_allowed("https://localhost:8000/foo") is False
    assert loader._is_url_allowed("http://127.0.0.1/admin") is False

    # Test 2: Explicit allowed_domains
    loader = GitbookLoader(
        web_page="https://docs.example.org/start",
        allowed_domains={"docs.example.org", "api.example.org"},
    )
    assert loader._is_url_allowed("https://docs.example.org/page") is True
    assert loader._is_url_allowed("https://api.example.org/v2/endpoint") is True
    assert loader._is_url_allowed("https://example.org/outside") is False
    assert loader._is_url_allowed("https://localhost:8000/foo") is False

    # Test 3: Disallowed initial URL
    with pytest.raises(ValueError):
        GitbookLoader(
            web_page="https://attacker.com/page", allowed_domains={"example.com"}
        )
