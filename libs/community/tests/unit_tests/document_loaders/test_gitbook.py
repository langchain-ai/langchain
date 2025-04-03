from typing import Any, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bs4 import BeautifulSoup
from langchain_core.documents import Document

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


@patch("langchain_community.document_loaders.web_base.requests.get")
def test_init_with_default_sitemap(mock_get: MagicMock) -> None:
    # Test that the loader uses the default sitemap URL when load_all_paths=True
    loader = GitbookLoader(web_page="https://example.com", load_all_paths=True)

    # Check that the web_path was set to the default sitemap URL
    assert loader.web_paths[0] == "https://example.com/sitemap.xml"


@patch("langchain_community.document_loaders.web_base.requests.get")
def test_init_with_custom_sitemap(mock_get: MagicMock) -> None:
    # Test that the loader uses the provided sitemap URL when specified
    custom_sitemap = "https://example.com/sitemap-pages.xml"
    loader = GitbookLoader(
        web_page="https://example.com",
        load_all_paths=True,
        sitemap_url=custom_sitemap,
    )

    # Check that the web_path was set to the custom sitemap URL
    assert loader.web_paths[0] == custom_sitemap


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


@patch("langchain_community.document_loaders.web_base.requests.get")
def test_with_single_page(mock_get: MagicMock) -> None:
    # Test loading a single page (load_all_paths=False)
    loader = GitbookLoader(web_page="https://example.com/page", load_all_paths=False)

    # Check that sitemap URL logic was not applied
    assert loader.web_paths[0] == "https://example.com/page"


@patch("langchain_community.document_loaders.gitbook.GitbookLoader.scrape")
def test_get_paths_extraction(
    mock_scrape: MagicMock, mock_soups: Tuple[BeautifulSoup, BeautifulSoup]
) -> None:
    # Test that _get_paths correctly extracts paths from sitemap
    mock_sitemap_soup, _ = mock_soups
    mock_scrape.return_value = mock_sitemap_soup

    loader = GitbookLoader(web_page="https://example.com", load_all_paths=True)

    soup_info = loader.scrape()
    paths = loader._get_paths(soup_info)

    # Check that paths were extracted correctly
    assert len(paths) == 3
    assert paths == ["/page1", "/page2", "/page3"]


@patch("requests.get")
def test_integration_with_different_sitemaps(mock_get: MagicMock) -> None:
    # This test simulates the reported issue with different sitemap formats

    # Mock response for default sitemap (empty content)
    empty_resp = MagicMock()
    empty_resp.text = "<urlset></urlset>"
    empty_resp.status_code = 200

    # Mock response for custom sitemap (with content)
    custom_resp = MagicMock()
    custom_resp.text = """
    <urlset>
        <url><loc>https://docs.gitbook.com/page1</loc></url>
        <url><loc>https://docs.gitbook.com/page2</loc></url>
    </urlset>
    """
    custom_resp.status_code = 200

    # Mock response for the actual pages
    page_resp = MagicMock()
    page_resp.text = """
    <html><body><main><h1>Page</h1><p>Content</p></main></body></html>
    """
    page_resp.status_code = 200

    # Define side effect to return different responses based on URL
    def side_effect(url: str, *args: Any, **kwargs: Any) -> MagicMock:
        if url == "https://docs.gitbook.com/sitemap.xml":
            return empty_resp
        elif url == "https://docs.gitbook.com/sitemap-pages.xml":
            return custom_resp
        else:
            return page_resp

    mock_get.side_effect = side_effect

    # Test with default sitemap (should result in no docs)
    with patch(
        "langchain_community.document_loaders.web_base.requests.get",
        side_effect=side_effect,
    ):
        with patch(
            "langchain_community.document_loaders.gitbook.GitbookLoader.scrape"
        ) as mock_scrape:
            with patch(
                "langchain_community.document_loaders.gitbook.GitbookLoader.scrape_all"
            ) as mock_scrape_all:
                mock_scrape.return_value = BeautifulSoup(
                    "<urlset></urlset>", "html.parser"
                )
                mock_scrape_all.return_value = []

                loader1 = GitbookLoader(
                    web_page="https://docs.gitbook.com/", load_all_paths=True
                )
                docs1 = list(loader1.lazy_load())
                assert len(docs1) == 0

    # Test with custom sitemap (should result in docs)
    with patch(
        "langchain_community.document_loaders.web_base.requests.get",
        side_effect=side_effect,
    ):
        with patch(
            "langchain_community.document_loaders.gitbook.GitbookLoader.scrape"
        ) as mock_scrape:
            with patch(
                "langchain_community.document_loaders.gitbook.GitbookLoader.scrape_all"
            ) as mock_scrape_all:
                mock_scrape.return_value = BeautifulSoup(
                    custom_resp.text, "html.parser"
                )
                mock_scrape_all.return_value = [
                    BeautifulSoup(page_resp.text, "html.parser"),
                    BeautifulSoup(page_resp.text, "html.parser"),
                ]

                loader2 = GitbookLoader(
                    web_page="https://docs.gitbook.com/",
                    load_all_paths=True,
                    sitemap_url="https://docs.gitbook.com/sitemap-pages.xml",
                )
                docs2 = list(loader2.lazy_load())
                assert len(docs2) == 2


@patch("langchain_community.document_loaders.gitbook.GitbookLoader._scrape")
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

    # Configure the mock to return different responses based on URL
    def scrape_side_effect(url: str, parser: str = "html.parser") -> BeautifulSoup:
        if url == "https://example.com/sitemap.xml":
            return mock_sitemap_index
        elif url == "https://example.com/sitemap-pages.xml":
            return mock_sitemap_pages
        elif url == "https://example.com/api/sitemap-pages.xml":
            return mock_sitemap_api_pages
        elif url == "https://example.com/changelog/sitemap-pages.xml":
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

    # Verify _scrape was called for the expected sitemap URLs
    # (parser doesn't matter now)
    expected_sitemap_urls = [
        "https://example.com/sitemap.xml",
        "https://example.com/sitemap-pages.xml",
        "https://example.com/api/sitemap-pages.xml",
        "https://example.com/changelog/sitemap-pages.xml",
    ]
    actual_scraped_urls = [call[0][0] for call in mock_scrape.call_args_list]
    assert len(actual_scraped_urls) == len(expected_sitemap_urls)
    assert set(actual_scraped_urls) == set(expected_sitemap_urls)

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


@patch("langchain_community.document_loaders.gitbook.GitbookLoader._scrape")
@patch("langchain_community.document_loaders.gitbook.GitbookLoader.scrape_all")
@pytest.mark.asyncio
async def test_aload_method(
    mock_scrape_all: MagicMock,
    mock_scrape: MagicMock,
) -> None:
    """Test the aload() method which returns a list of documents asynchronously."""
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

    # Create mock for async methods
    mock_doc = Document(
        page_content="This is test content.",
        metadata={"source": "test", "title": "Test Page"},
    )

    async def mock_afetch(url: str) -> Document:
        return mock_doc

    # Create loader and patch async methods
    with (
        patch.object(
            GitbookLoader,
            "_aprocess_sitemap",
            new_callable=AsyncMock,
            return_value=["https://example.com/page1", "https://example.com/page2"],
        ),
        patch.object(
            GitbookLoader,
            "_afetch_and_process_url",
            new_callable=AsyncMock,
            side_effect=mock_afetch,
        ),
    ):
        loader = GitbookLoader(
            web_page="https://example.com",
            load_all_paths=True,
        )

        # Test the aload() method (async)
        docs = await loader.aload()

        # Verify results
        assert isinstance(docs, list)
        assert len(docs) == 2
        for doc in docs:
            assert "This is test content." in doc.page_content
            assert doc.metadata["title"] == "Test Page"


@pytest.mark.asyncio
async def test_alazy_load_method() -> None:
    """Test the alazy_load() async generator method."""
    # Create mock document
    mock_doc = Document(
        page_content="This is test content.",
        metadata={"source": "test", "title": "Test Page"},
    )

    # Define async mock functions
    async def mock_ascrape(url: str, parser: Optional[str] = None) -> BeautifulSoup:
        sitemap_content = """
        <urlset>
            <url><loc>https://example.com/page1</loc></url>
            <url><loc>https://example.com/page2</loc></url>
        </urlset>
        """
        return BeautifulSoup(sitemap_content, "html.parser")

    async def mock_aprocess_sitemap(soup: BeautifulSoup, base_url: str) -> list[str]:
        return ["https://example.com/page1", "https://example.com/page2"]

    async def mock_afetch(url: str) -> Document:
        return mock_doc

    # Patch the async methods
    with (
        patch.object(
            GitbookLoader, "_ascrape", new_callable=AsyncMock, side_effect=mock_ascrape
        ),
        patch.object(
            GitbookLoader,
            "_aprocess_sitemap",
            new_callable=AsyncMock,
            side_effect=mock_aprocess_sitemap,
        ),
        patch.object(
            GitbookLoader,
            "_afetch_and_process_url",
            new_callable=AsyncMock,
            side_effect=mock_afetch,
        ),
    ):
        # Create loader
        loader = GitbookLoader(
            web_page="https://example.com",
            load_all_paths=True,
        )

        # Test the alazy_load() method (async generator)
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)

        # Verify results
        assert len(docs) == 2
        for doc in docs:
            assert "This is test content." in doc.page_content
            assert doc.metadata["title"] == "Test Page"


@patch("langchain_community.document_loaders.gitbook.GitbookLoader._scrape")
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
